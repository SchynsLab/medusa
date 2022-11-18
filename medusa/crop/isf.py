""" Crop model adapted from the insightface implementation. By reimplementing
it here, insightface does not have to be installed.

Please see the LICENSE file in the current directory for the license that
is applicable to this implementation.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from onnxruntime import InferenceSession
from onnxruntime import set_default_logger_severity
from skimage.transform import estimate_transform, warp

from .base import BaseCropModel
        

class InsightfaceCropModel(BaseCropModel):
    """ Cropping model based on functionality from the ``insightface`` package, as used
    by MICA (https://github.com/Zielon/MICA).

    Parameters
    ----------
    name : str
        Name of underlying insightface model
    det_size : tuple
        Image size for detection
    target_size : tuple
        Length 2 tuple with desired width/heigth of cropped image; should be (112, 112)
        for MICA
    det_thresh : float
        Detection threshold (higher = more stringent)
    device : str
        Either 'cuda' (GPU) or 'cpu'

    Examples
    --------
    To crop an image to be used for MICA reconstruction:

    >>> from medusa.data import get_example_frame
    >>> crop_model = InsightfaceCropModel(device='cpu')
    >>> img = get_example_frame()  # path to jpg image
    >>> crop_img = crop_model(img)
    >>> crop_img.shape
    torch.Size([1, 3, 112, 112])
    """    

    def __init__(self, det_thresh=0.5, nms_thresh=0.4, input_size=(224, 224), return_bbox=False, device='cuda'):
        
        self.nms_thresh = det_thresh
        self.det_thresh = nms_thresh
        self.input_size = input_size
        self.return_bbox = return_bbox
        self.device = device
        self.session = self._init_session()        
        self.center_cache = {}
        self._mean = 127.5
        self._scale = 128
        self._to_bgr = True
        self._init_vars()

    def _init_session(self):

        f_in = Path.home() / '.insightface/models/buffalo_l/det_10g.onnx'
        if not f_in.is_file():
            import gdown
            f_out = Path.home() / '.insightface/models/buffalo_l.zip'
            if not f_out.parent.is_dir():
                f_out.parent.mkdir(parents=True)
            
            # alternative: #http://insightface.cn-sh2.ufileos.com/models/buffalo_l.zip
            url = 'https://drive.google.com/u/0/uc?id=1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB&export=download'
            gdown.download(url, str(f_out), )
            gdown.extractall(str(f_out))

        set_default_logger_severity(3)
        return InferenceSession(str(f_in), providers=['CPUExecutionProvider'])

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._anchor_ratio = 1.0
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        
    def _forward(self, img, threshold):

        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name : blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=threshold)[0]
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            kpss = self._distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)
    
        return scores_list, bboxes_list, kpss_list

    def _detect(self, img, max_num=0, metric='default'):
        
        input_size = self.input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self._forward(det_img, self.det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self._nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        
        return det, kpss

    def _nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def _distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i%2] + distance[:, i]
            py = points[:, i%2+1] + distance[:, i+1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def __call__(self, images):
            
        images = self._load_inputs(images, to_bgr=self._to_bgr, channels_first=False, load_as='numpy')

        img_crop = np.zeros((images.shape[0], 112, 112, 3))
        bbox = np.zeros((images.shape[0], 4, 2))
        crop_mat = np.zeros((images.shape[0], 3, 3))

        for i in range(images.shape[0]):
            image = images[i, ...]
            bboxes, kpss = self._detect(image, max_num=0, metric='default')
            i = self._get_center(bboxes, image)
            bbox_ = bboxes[i, 0:4]
            bbox[i, ...] = np.array([
                [bbox_[0], bbox_[1]],
                [bbox_[2], bbox_[1]],
                [bbox_[0], bbox_[2]],
                [bbox_[2], bbox_[3]]
            ])
            #det_score = bboxes[i, 4]
            
            kps = None
            if kpss is not None:
                kps = kpss[i]
            
            # Crop to target size using keypoints (kps)
            src = np.array(
                [
                    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]
                ], dtype=np.float32
            )
            crop_mat_ = estimate_transform("similarity", kps, src)
            img_crop[i, ...] = warp(image, crop_mat_.inverse, output_shape=(112, 112), preserve_range=True)
            crop_mat[i, ...] = crop_mat_.params

        # BGR -> RGB, because MICA needs RGB
        img_crop = img_crop[:, :, :, ::-1]
        
        # Channel-wise mean subtraction (- 127.5), scaling (* 1 / 127.5) 
        img_crop = ((img_crop - self._mean) / self._scale)
 
        # Channels first
        img_crop = img_crop.transpose(0, 3, 1, 2)
        img_crop = torch.from_numpy(img_crop).to(self.device)

        crop_mat = torch.from_numpy(crop_mat).to(self.device)

        if self.return_bbox:
            return img_crop, crop_mat, bbox
        else:
            return img_crop, crop_mat

    def _dist(self, p1, p2):
        return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    
    def _get_center(self, bboxes, img):
        
        img_center = img.shape[0] // 2, img.shape[1] // 2
        size = bboxes.shape[0]
        distance = np.Inf
        j = 0
        for i in range(size):
            x1, y1, x2, y2 = bboxes[i, 0:4]
            dx = abs(x2 - x1) / 2.0
            dy = abs(y2 - y1) / 2.0
            current = self._dist((x1 + dx, y1 + dy), img_center)
            if current < distance:
                distance = current
                j = i

        return j

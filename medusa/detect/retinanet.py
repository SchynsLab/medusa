""" Face detection model adapted from the insightface implementation. By reimplementing
it here, insightface does not have to be installed.

Please see the LICENSE file in the current directory for the license that
is applicable to this implementation.
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from kornia.geometry.transform import resize
from onnxruntime import InferenceSession
from onnxruntime import set_default_logger_severity

from .. import DEVICE
from ..io import load_inputs
from .base import BaseDetectionModel, DetectionResults


class RetinanetDetector(BaseDetectionModel):
    """ Face detection model based on the ``insightface`` package, as used
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

    def __init__(self, det_threshold=0.5, nms_threshold=0.3, device=DEVICE):
        
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self._det_model = self._init_det_model()        
        self._center_cache = {}
        self._mean = 127.5
        self._scale = 128
        self._init_vars()

    def __str__(self):
        return 'retinanet'

    def _init_det_model(self):
        
        f_in = Path(__file__).parents[1] / 'data/models/buffalo_l/det_10g.onnx'
        #f_in = Path(__file__).parents[1] / 'data/models/antelopev2/scrfd_10g_bnkps.onnx'
        
        if not f_in.is_file():
            f_out = f_in.parents[1] / 'buffalo_l.zip'

            # alternative: #http://insightface.cn-sh2.ufileos.com/models/buffalo_l.zip
            import gdown
            url = 'https://drive.google.com/u/0/uc?id=1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB&export=download'
            gdown.download(url, str(f_out))
            gdown.extractall(str(f_out))

            f_out.unlink()

        set_default_logger_severity(3)
        device = self.device.upper()
        sess = InferenceSession(str(f_in), providers=[f'{device}ExecutionProvider'])
        self._onnx_input_name = sess.get_inputs()[0].name
        self._onnx_input_shape = (3, 224, 224)  # undefined in this onnx model
        self._onnx_output_names = [o.name for o in sess.get_outputs()]

        self.binding = sess.io_binding()
        for name in self._onnx_output_names:
            self.binding.bind_output(name)

        return sess 

    def _init_vars(self):
        self._anchor_ratio = 1.0
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        
    def _forward(self, det_img):

        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        self.binding.bind_input(
            name=self._onnx_input_name,
            device_type=self.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(det_img.shape),
            buffer_ptr=det_img.data_ptr(),
        )

        self._det_model.run_with_iobinding(self.binding)
        net_outputs = self.binding.copy_outputs_to_cpu()
        
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            
            scores = net_outputs[idx]
            bbox_preds = net_outputs[idx+fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = net_outputs[idx+fmc*2] * stride
            
            height = 224 // stride
            width = 224 // stride
            K = height * width
            key = (height, width, stride)
            if key in self._center_cache:
                anchor_centers = self._center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self._center_cache)<100:
                    self._center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_threshold)[0]
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

    def __call__(self, imgs, max_num=0):
        
        # B x C x H x W        
        imgs = load_inputs(imgs, load_as='torch', channels_first=True, device=self.device)
        b, c, h, w = imgs.shape
        im_ratio = float(h) / w

        # model_ratio = 1.
        _, mh, mw = self._onnx_input_shape
        model_ratio = float(mh) / mw
        
        if im_ratio > model_ratio:
            new_h = mh
            new_w = int(new_h / im_ratio)
        else:
            new_w = mw
            new_h = int(new_w * im_ratio)
                
        imgs = resize(imgs, (new_h, new_w))
        det_imgs = torch.zeros((b, c, mh, mw), device=imgs.device)
        det_imgs[:, :, :new_h, :new_w] = imgs
        det_imgs = ((det_imgs - self._mean) / self._scale)[:, [2, 1, 0], :, :]
        det_scale = float(new_h) / h

        outputs = defaultdict(list)
        # Loop over batch, because this onnx model does not support batch
        # inference
        for i in range(b):
            det_img = det_imgs[i, ...].unsqueeze(0)  # 1 x 3 x 224 x 224
            scores_list, bboxes_list, kpss_list = self._forward(det_img)
            
            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            bboxes = np.vstack(bboxes_list) / det_scale
            kpss = np.vstack(kpss_list) / det_scale
            
            pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
            pre_det = pre_det[order, :]
            keep = self._nms(pre_det)
            det = pre_det[keep, :]
            
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
            
            if max_num > 0 and det.shape[0] > max_num:
                area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = h // 2, w // 2
                offsets = np.vstack([
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]
                ])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
                bindex = np.argsort(values)[::-1]  # some extra weight on the centering
                bindex = bindex[0:max_num]
                det = det[bindex, :]
                kpss = kpss[bindex, :]

            if len(det):
                outputs['img_idx'].extend([[i] * det.shape[0]])
                outputs['conf'].append(det[:, 4])
                outputs['bbox'].append(det[:, :4])
                outputs['lms'].append(kpss)
        
        outputs = DetectionResults(imgs.shape[0], **outputs, device=self.device)
        return outputs

    def _nms(self, dets):
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

            inds = np.where(ovr <= self.nms_threshold)[0]
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

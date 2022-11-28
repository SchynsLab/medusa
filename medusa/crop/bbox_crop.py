import torch
import numpy as np
from pathlib import Path
from kornia.geometry.transform import warp_affine
from kornia.geometry.linalg import transform_points
from onnxruntime import set_default_logger_severity, InferenceSession

from .. import DEVICE
from .base import BaseCropModel, CropResults
from ..io import load_inputs
from ..detect import RetinanetDetector
from ..transforms import estimate_similarity_transform


class LandmarkBboxCropModel(BaseCropModel):
    def __init__(self, name='2d106det', output_size=(224, 244), detector=RetinanetDetector,
                 return_lmk=False, device=DEVICE, **kwargs):
        # alternative: 1k3d68, 2d106det
        self.name = name
        self.output_size = output_size  # h, w
        self._detector = detector(device=device, **kwargs)
        self.return_lmk = return_lmk
        self.device = device
        self._lmk_model = self._init_lms_model()

    def _init_lms_model(self):

        set_default_logger_severity(3)
        
        f_in = Path(__file__).parents[1] / f'data/models/buffalo_l/{self.name}.onnx'
        device = self.device.upper()
        sess = InferenceSession(str(f_in), providers=[f'{device}ExecutionProvider'])
        self.input_name = sess.get_inputs()[0].name
        self.input_shape = sess.get_inputs()[0].shape
        self.output_names = [o.name for o in sess.get_outputs()]
        self.output_shape = [o.shape for o in sess.get_outputs()]
        
        self.binding = sess.io_binding()
        for name in self.output_names:
            self.binding.bind_output(name)

        return sess

    def __str__(self):
        return 'bboxcrop'

    def _create_bbox(self, lm, scale=1.25):
        """ Creates a bounding box (bbox) based on the landmarks by creating
        a box around the outermost landmarks (+10%), as done in the original
        DECA usage.

        Parameters
        ----------
        scale : float
            Factor to scale the bounding box with
        """
        left = torch.min(lm[:, :, 0], dim=1)[0]
        right = torch.max(lm[:, :, 0], dim=1)[0]
        top = torch.min(lm[:, :, 1], dim=1)[0]
        bottom = torch.max(lm[:, :, 1], dim=1)[0]

        # scale and 1.1 are DECA constants
        orig_size = (right - left + bottom - top) / 2 * 1.1
        size = (orig_size * scale)  # to int?

        # b x 2 (center coords)    
        center = torch.stack([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0], dim=1)
        
        b = lm.shape[0]
        bbox = torch.zeros((b, 4, 2), device=self.device)
        bbox[:, 0, :] = center - size[:, None] / 2
        bbox[:, 1, 0] = center[:, 0] - size / 2
        bbox[:, 1, 1] = center[:, 1] + size / 2
        bbox[:, 2, 0] = center[:, 0] + size / 2
        bbox[:, 2, 1] = center[:, 1] - size / 2
        bbox[:, 3, :] = center + size[:, None] / 2

        return bbox

    def __call__(self, images):
        # Load images here instead of in detector to avoid loading them twice

        imgs = load_inputs(images, load_as='torch', channels_first=True, device=self.device)
        out_det = self._detector(imgs)
        
        n_det = len(out_det)
        if n_det == 0:
            return CropResults(None, None, None, None, self.device)
        
        bbox = out_det.bbox
        imgs_stack = imgs[out_det.idx]

        #b, c, w, h = imgs_stk.shape
        bw, bh = (bbox[:, 2] - bbox[:, 0]), (bbox[:, 3] - bbox[:, 1])
        center = torch.stack([(bbox[:, 2] + bbox[:, 0]) / 2,
                              (bbox[:, 3] + bbox[:, 1]) / 2], dim=1)
        
        scale = self.input_shape[3] / (torch.maximum(bw, bh) * 1.5)

        M = torch.eye(3, device=self.device).repeat(n_det, 1, 1) * scale[:, None, None]
        M[:, 2, 2] = 1
        M[:, :2, 2] = -1 * center * scale[:, None] + self.input_shape[3] / 2
        imgs_crop = warp_affine(imgs_stack, M[:, :2, :], dsize=self.input_shape[2:])

        self.binding.bind_input(
            name=self.input_name,
            device_type=self.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(imgs_crop.shape),
            buffer_ptr=imgs_crop.data_ptr(),
        )
        
        lms_shape = (n_det, self.output_shape[0][1])
        lms = torch.empty(lms_shape, dtype=torch.float32, device=self.device)
        self.binding.bind_output(
            name=self.output_names[0],
            device_type=self.device,
            device_id=0,
            element_type=np.float32,
            shape=lms_shape,
            buffer_ptr=lms.data_ptr(),
        )

        self._lmk_model.run_with_iobinding(self.binding)
        
        if lms.shape[1] == 3309:
            lms = lms.reshape((n_det, -1, 3))
            lms = lms[:, -68:, :]
        else:
            lms = lms.reshape((n_det, -1, 2))
            
        lms[:, :, 0:2] = (lms[:, :, 0:2] + 1) * (self.input_shape[3] // 2)
        if lms.shape[2] == 3:
            lms[:, :, 2] *= (self.input_shape[3] // 2)

        lms = lms[:, :, :2]  # don't need 3rd dim
        lms = transform_points(torch.inverse(M), lms)
        
        bbox = self._create_bbox(lms)
        
        w, h = self.output_size
        dst = torch.tensor([[0, 0], [0, w-1], [h-1, 0]], dtype=torch.float32, device=self.device)
        dst = dst.repeat(n_det, 1, 1)

        crop_mats = estimate_similarity_transform(bbox[:, :3, :], dst, estimate_scale=True)
        imgs_crop = warp_affine(imgs_stack, crop_mats[:, :2, :], dsize=(h, w))
        lms = transform_points(crop_mats, lms)

        out_crop = CropResults(imgs_crop, crop_mats, lms, out_det.idx, device=self.device)

        return out_crop
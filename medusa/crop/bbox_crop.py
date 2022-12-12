from pathlib import Path

import numpy as np
import torch
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import warp_affine
from onnxruntime import InferenceSession, set_default_logger_severity

from .. import DEVICE
from ..detect import SCRFDetector
from ..io import load_inputs
from ..transforms import estimate_similarity_transform
from .base import BaseCropModel, CropResults


class LandmarkBboxCropModel(BaseCropModel):
    def __init__(self, name='2d106det', output_size=(224, 224), detector=SCRFDetector,
                 device=DEVICE, **kwargs):
        # alternative: 1k3d68, 2d106det
        self.name = name
        self.output_size = output_size  # h, w
        self._detector = detector(device=device, **kwargs)
        self.device = device
        self._lms_model = self._init_lms_model()

    def _init_lms_model(self):

        set_default_logger_severity(3)

        f_in = Path(__file__).parents[1] / f'data/models/buffalo_l/{self.name}.onnx'
        device = self.device.upper()
        opts = {"cudnn_conv_algo_search": "HEURISTIC"}
        sess = InferenceSession(str(f_in), providers=[(f'{device}ExecutionProvider', opts)])
        self._onnx_input_name = sess.get_inputs()[0].name
        self._onnx_input_shape = sess.get_inputs()[0].shape
        self._onnx_output_names = [o.name for o in sess.get_outputs()]
        self._onnx_output_shape = [o.shape for o in sess.get_outputs()]

        return sess

    def __str__(self):
        return 'bboxcrop'

    def _create_bbox(self, lm, scale=1.25):
        """Creates a bounding box (bbox) based on the landmarks by creating a
        box around the outermost landmarks (+10%), as done in the original DECA
        usage.

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
            return CropResults(imgs.shape[0], device=self.device)

        bbox = out_det.bbox
        imgs_stack = imgs[out_det.img_idx]

        bw, bh = (bbox[:, 2] - bbox[:, 0]), (bbox[:, 3] - bbox[:, 1])
        center = torch.stack([(bbox[:, 2] + bbox[:, 0]) / 2,
                              (bbox[:, 3] + bbox[:, 1]) / 2], dim=1)

        scale = self._onnx_input_shape[3] / (torch.maximum(bw, bh) * 1.5)

        # Construct initial crop matrix based on rough bounding box,
        # then crop images to size 192 x 192
        M = torch.eye(3, device=self.device).repeat(n_det, 1, 1) * scale[:, None, None]
        M[:, 2, 2] = 1
        M[:, :2, 2] = -1 * center * scale[:, None] + self._onnx_input_shape[3] / 2
        imgs_crop = warp_affine(imgs_stack, M[:, :2, :], dsize=self._onnx_input_shape[2:]).contiguous()

        # Note to self: not necessary to normalize data (as per insightface's implementation)
        binding = self._lms_model.io_binding()
        binding.bind_input(
            name=self._onnx_input_name,
            device_type=self.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(imgs_crop.shape),
            buffer_ptr=imgs_crop.data_ptr(),
        )

        lms_shape = (n_det, self._onnx_output_shape[0][1])
        lms = torch.empty(lms_shape, dtype=torch.float32, device=self.device).contiguous()
        binding.synchronize_outputs()  # idk why, but *need* to call this otherwise lms is not updated
        binding.bind_output(
            name=self._onnx_output_names[0],
            device_type=self.device,
            device_id=0,
            element_type=np.float32,
            shape=lms_shape,
            buffer_ptr=lms.data_ptr(),
        )
        self._lms_model.run_with_iobinding(binding)

        if lms.shape[1] == 3309:
            lms = lms.reshape((n_det, -1, 3))
            lms = lms[:, -68:, :]
        else:
            lms = lms.reshape((n_det, -1, 2))

        lms[:, :, 0:2] = (lms[:, :, 0:2] + 1) * (self._onnx_input_shape[3] // 2)
        if lms.shape[2] == 3:
            lms[:, :, 2] *= (self._onnx_input_shape[3] // 2)

        lms = lms[:, :, :2]  # don't need 3rd dim

        # Warp landmarks from initial crop space (192 x 192) to
        # the original image space (?, ?), and create a new
        # "DECA style" bbox
        lms = transform_points(torch.inverse(M), lms)
        bbox = self._create_bbox(lms)

        # Estimate a transform from the new bbox to the final
        # cropped image space (probably 224 x 224 for DECA-based models)
        w, h = self.output_size
        dst = torch.tensor([[0, 0], [0, w-1], [h-1, 0]], dtype=torch.float32, device=self.device)
        dst = dst.repeat(n_det, 1, 1)
        crop_mats = estimate_similarity_transform(bbox[:, :3, :], dst, estimate_scale=True)

        # Finally, warp the original images (uncropped) images to the final
        # cropped space
        imgs_crop = warp_affine(imgs_stack, crop_mats[:, :2, :], dsize=(h, w))
        out_crop = CropResults(imgs.shape[0], imgs_crop, crop_mats, lms, out_det.img_idx, out_det.face_idx, device=self.device)

        return out_crop

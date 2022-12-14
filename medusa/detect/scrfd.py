"""SCRFD face detection model adapted from the insightface implementation. All
numpy code has been converted to torch, speeding it up considerably (especially
with GPU and large batch size). Please cite the corresponding paper `[1]` from
the people at insightface if you use this implementation.

Also, lease see the LICENSE file in the current directory for the
license that is applicable to this implementation.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from onnxruntime import (InferenceSession,
                         set_default_logger_severity)
from torchvision.ops import nms

from .. import DEVICE
from ..io import load_inputs
from ..transforms import resize_with_pad
from .base import BaseDetectionModel


class SCRFDetector(BaseDetectionModel):
    """Face detection model based on the ``insightface`` package, as used by
    MICA (https://github.com/Zielon/MICA).

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

    def __str__(self):
        return 'scrfd'

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

        # per: https://medium.com/neuml/debug-onnx-gpu-performance-c9290fe07459
        opts = {"cudnn_conv_algo_search": "HEURISTIC"}
        sess = InferenceSession(str(f_in), providers=[(f'{device}ExecutionProvider', opts)])
        self._onnx_input_name = sess.get_inputs()[0].name
        self._onnx_input_shape = (1, 3, 224, 224)  # undefined in this onnx model
        self._onnx_output_names = [o.name for o in sess.get_outputs()]

        return sess

    def __call__(self, imgs):

        # B x C x H x W
        imgs = load_inputs(imgs, load_as='torch', channels_first=True, device=self.device)
        b, c, h, w = imgs.shape

        new_size = self._onnx_input_shape[-2:]  # 224 x 224
        imgs, det_scale = resize_with_pad(imgs, output_size=new_size, out_dtype=torch.float32)
        imgs = (imgs.sub_(127.5)).div_(128)

        outputs = defaultdict(list)
        for i in range(b):
            # add batch dim
            img = imgs[i, ...].unsqueeze(0)

            binding = self._det_model.io_binding()
            binding.bind_input(
                name=self._onnx_input_name,
                device_type=self.device,
                device_id=0,
                element_type=np.float32,
                shape=tuple(img.shape),
                buffer_ptr=img.data_ptr(),
            )

            feat_stride_fpn = [8, 16, 32]
            net_outputs = []
            for n_feat in [1, 4, 10]:  # score, bbox, lms
                for stride in feat_stride_fpn:
                    out_shape = ((224 // stride) ** 2 * 2, n_feat)
                    out = torch.empty(out_shape, dtype=torch.float32, device=self.device).contiguous()
                    net_outputs.append(out)

            for i_out, out_name in enumerate(self._onnx_output_names):

                binding.bind_output(
                    name=out_name,
                    device_type=self.device,
                    device_id=0,
                    element_type=np.float32,
                    shape=tuple(net_outputs[i_out].shape),
                    buffer_ptr=net_outputs[i_out].data_ptr(),
                )

            binding.synchronize_inputs
            binding.synchronize_outputs()
            self._det_model.run_with_iobinding(binding)

            fmc = 3
            num_anchors = 2

            outputs_ = defaultdict(list)
            for idx, stride in enumerate(feat_stride_fpn):
                scores = net_outputs[idx].flatten()  # (n_det,)
                bbox = net_outputs[idx+fmc] * stride  # (n_det, 4)
                lms = net_outputs[idx+fmc*2] * stride  # (n_det, 10)
                keep = torch.where(scores >= self.det_threshold)[0]

                if len(keep) == 0:
                    continue

                scores = scores[keep]
                bbox = bbox[keep]
                lms = lms[keep]

                # (h, w, 2) -> (h * w, 2) -> (h * w * 2, 2)
                grid = torch.arange(0, 224 // stride, device=self.device)
                anchor_centers = torch.stack(torch.meshgrid(grid, grid, indexing='xy'), dim=-1)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = torch.stack([anchor_centers] * num_anchors, dim=1).reshape((-1, 2))
                anchor_centers = anchor_centers[keep]

                # Distance-to-bbox/lms
                bbox = torch.hstack([
                    anchor_centers[:, :2] - bbox[:, :2],
                    anchor_centers[:, [0, 1]] + bbox[:, [2, 3]]
                ])
                lms = anchor_centers[:, None, :] + lms.reshape((lms.shape[0], -1, 2))
                outputs_['scores'].append(scores)
                outputs_['bbox'].append(bbox)
                outputs_['lms'].append(lms)

            if len(outputs_['scores']) == 0:
                continue

            scores = torch.cat(outputs_['scores'])
            bbox = torch.vstack(outputs_['bbox']) / det_scale
            lms = torch.vstack(outputs_['lms']) / det_scale
            keep = nms(bbox, scores, self.nms_threshold)

            n_keep = len(keep)
            if n_keep > 0:
                img_idx = torch.ones(n_keep, device=self.device, dtype=torch.int64) * i
                outputs['img_idx'].append(img_idx)
                outputs['conf'].append(scores[keep])
                outputs['lms'].append(lms[keep])
                outputs['bbox'].append(bbox[keep])

        for attr, data in outputs.items():
            outputs[attr] = torch.cat(data)

        outputs['n_img'] = b
        return outputs

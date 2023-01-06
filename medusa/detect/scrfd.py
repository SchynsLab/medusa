"""SCRFD face detection model adapted from the insightface implementation. All
numpy code has been converted to torch, speeding it up considerably (especially
with GPU and large batch size). Please cite the corresponding paper `[1]` from
the people at insightface if you use this implementation.

Also, lease see the LICENSE file in the current directory for the
license that is applicable to this implementation.
"""

from collections import defaultdict

import torch
from torchvision.ops import nms

from .base import BaseDetector
from ..defaults import DEVICE
from ..io import load_inputs
from ..data import get_external_data_config
from ..onnx import OnnxModel
from ..transforms import resize_with_pad


class SCRFDetector(BaseDetector):
    """Face detection model based on the ``insightface`` package.

    Parameters
    ----------
    name : str
        Name of underlying insightface model
    det_size : tuple
        Size to which the input image(s) will be resized before passing it to the
        detection model; should be a tuple with two of the same integers (indicating
        a square image); higher values are more accurate but slower; default is
        (224, 224)
    det_threshold : float
        Detection threshold (higher = more conservative); detections with confidence
        values lower than ``det_threshold`` are discarded
    nms_threshold : float
        Non-maximum suppression threshold for predicted bounding boxes
    device : str
        Either 'cuda' (GPU) or 'cpu'

    Examples
    --------
    To crop an image to be used for MICA reconstruction:

    >>> from medusa.data import get_example_frame
    >>> det_model = SCRFDetector()
    >>> img = get_example_frame()  # path to jpg image
    >>> det_results = det_model(img)
    >>> list(det_results.keys())
    ['img_idx', 'conf', 'lms', 'bbox', 'n_img']
    """

    def __init__(self, det_size=(224, 224), det_threshold=0.5, nms_threshold=0.3,
                 device=DEVICE):

        self.det_size = det_size
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self._det_model = self._init_det_model()

    def __str__(self):
        return "scrfd"

    def _init_det_model(self):

        if self.det_size[0] != self.det_size[1]:
            return ValueError("Param ``det_size`` should indicate a square image, "
                             f"e.g., (224, 224), but got {self.det_size}!")
        h = self.det_size[0]

        f_in = get_external_data_config('buffalo_path') / 'det_10g.onnx'
        if not f_in.is_file():
            raise ValueError("Insightface model needs to be downloaded; run `medusa_download_external_data`")

        output_shapes = []
        for n_feat in [1, 4, 10]:  # score, bbox, lms
            for stride in [8, 16, 32]:
                output_shapes.append(((h // stride) ** 2 * 2, n_feat))

        return OnnxModel(
            f_in,
            device=self.device,
            in_shapes=[[1, 3, *self.det_size]],
            out_shapes=output_shapes,
        )

    def __call__(self, imgs):

        # B x C x H x W
        imgs = load_inputs(
            imgs, load_as="torch", channels_first=True, device=self.device
        )
        b = imgs.shape[0]  # batch size

        imgs, det_scale = resize_with_pad(
            imgs, output_size=self.det_size, out_dtype=torch.float32
        )
        imgs = (imgs.sub_(127.5)).div_(128)  # normalize for onnx model

        outputs = defaultdict(list)
        for i in range(b):
            # add batch dim
            img = imgs[i, ...].unsqueeze(0)
            det_outputs = self._det_model.run(img, outputs_as_list=True)

            fmc, num_anchors = 3, 2
            feat_stride_fpn = [8, 16, 32]
            outputs_ = defaultdict(list)
            for idx, stride in enumerate(feat_stride_fpn):
                scores = det_outputs[idx].flatten()  # (n_det,)
                bbox = det_outputs[idx + fmc] * stride  # (n_det, 4)
                lms = det_outputs[idx + fmc * 2] * stride  # (n_det, 10)
                keep = torch.where(scores >= self.det_threshold)[0]

                if len(keep) == 0:
                    continue

                scores = scores[keep]
                bbox = bbox[keep]
                lms = lms[keep]

                # (h, w, 2) -> (h * w, 2) -> (h * w * 2, 2)
                grid = torch.arange(0, self.det_size[0] // stride, device=self.device)
                anchor_centers = torch.stack(
                    torch.meshgrid(grid, grid, indexing="xy"), dim=-1
                )
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = torch.stack(
                    [anchor_centers] * num_anchors, dim=1
                ).reshape((-1, 2))
                anchor_centers = anchor_centers[keep]

                # Distance-to-bbox/lms
                bbox = torch.hstack(
                    [
                        anchor_centers[:, :2] - bbox[:, :2],
                        anchor_centers[:, [0, 1]] + bbox[:, [2, 3]],
                    ]
                )
                lms = anchor_centers[:, None, :] + lms.reshape((lms.shape[0], -1, 2))
                outputs_["scores"].append(scores)
                outputs_["bbox"].append(bbox)
                outputs_["lms"].append(lms)

            if len(outputs_["scores"]) == 0:
                continue

            scores = torch.cat(outputs_["scores"])
            bbox = torch.vstack(outputs_["bbox"]) / det_scale
            lms = torch.vstack(outputs_["lms"]) / det_scale
            keep = nms(bbox, scores, self.nms_threshold)

            n_keep = len(keep)
            if n_keep > 0:
                img_idx = torch.ones(n_keep, device=self.device, dtype=torch.int64) * i
                outputs["img_idx"].append(img_idx)
                outputs["conf"].append(scores[keep])
                outputs["lms"].append(lms[keep])
                outputs["bbox"].append(bbox[keep])

        for attr, data in outputs.items():
            outputs[attr] = torch.cat(data)

        outputs["n_img"] = b
        return outputs

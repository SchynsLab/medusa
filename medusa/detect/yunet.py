from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..defaults import DEVICE
from ..io import load_inputs
from .base import BaseDetector


class YunetDetector(BaseDetector):
    """This detector is based on Yunet, a face detector based on YOLOv3 :cite:p:`facedetect-yu`."""
    def __init__(self, det_threshold=0.9, nms_threshold=0.3, device=DEVICE, **kwargs):
        super().__init__()
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self._model = self._init_model(**kwargs)
        self.to(device).eval()

    def __str__(self):
        return "yunet"

    def _init_model(self, **kwargs):

        try:
            import cv2
        except ImportError:
            raise ImportError("cv2 is required for YunetDetector")

        f_in = Path(__file__).parents[1] / "data/models/yunet.onnx"
        model = cv2.FaceDetectorYN.create(
            str(f_in), "", (0, 0), self.det_threshold, self.nms_threshold, **kwargs
        )

        return model

    def forward(self, imgs):

        imgs = load_inputs(imgs, load_as="numpy", channels_first=False)
        # cv2 needs BGR (not RGB)
        imgs = imgs[:, :, :, [2, 0, 1]]

        b, h, w, c = imgs.shape
        self._model.setInputSize((w, h))
        outputs = defaultdict(list)

        # Note to self: cv2 does not support batch prediction
        for i in range(b):

            _, det = self._model.detect(imgs[i, ...])

            if det is not None:
                outputs["img_idx"].extend([i] * det.shape[0])
                outputs["conf"].append(det[:, [-1]].flatten())
                bbox_ = det[:, :4]
                # Convert offset to true vertex positions to keep consistent
                # with scrfd/torchvision bbox definition
                bbox_[:, 2:] = bbox_[:, :2] + bbox_[:, 2:]
                outputs["bbox"].append(bbox_)
                outputs["lms"].append(det[:, 4:-1].reshape((det.shape[0], 5, 2)))

        if outputs.get("conf", None) is not None:
            outputs["img_idx"] = np.array(outputs["img_idx"])
            outputs["conf"] = np.concatenate(outputs["conf"])
            outputs["bbox"] = np.vstack(outputs["bbox"])
            outputs["lms"] = np.vstack(outputs["lms"])
            for attr, data in outputs.items():
                outputs[attr] = torch.as_tensor(data, device=self.device)

        outputs["n_img"] = b

        return outputs

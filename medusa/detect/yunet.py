import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from .. import DEVICE
from .base import BaseDetectionModel
from ..io import load_inputs


class YunetDetector(BaseDetectionModel):

    def __init__(self, det_threshold=0.9, nms_threshold=0.3, device=DEVICE, **kwargs):

        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self._model = self._init_model(**kwargs)
    
    def __str__(self):
        return 'yunet'

    def _init_model(self, **kwargs):

        f_in = Path(__file__).parents[1] / 'data/models/yunet.onnx'
        model = cv2.FaceDetectorYN.create(
            str(f_in), '', (0, 0),
            self.det_threshold, self.nms_threshold,
            **kwargs
        )

        return model
    
    def __call__(self, imgs):

        imgs = load_inputs(imgs, load_as='numpy', channels_first=False)
        # cv2 needs BGR (not RGB)
        imgs = imgs[:, :, :, [2, 0, 1]]

        b, h, w, c = imgs.shape
        self._model.setInputSize((w, h))
        outputs = defaultdict(list)

        # Note to self: cv2 does not support batch prediction
        for i in range(b):

            _, det = self._model.detect(imgs[i, ...])

            if det is None:
                outputs['idx'].append([[np.nan]])
                outputs['conf'].append([[np.nan]])
                outputs['bbox'].append(np.full((1, 4), np.nan))
                outputs['lms'].append(np.full((1, 4), np.nan))
            else:
                outputs['conf'].append(det[:, -1])
                bbox_ = det[:, :4]
                # Convert offset to true vertex positions to keep consistent
                # with retinanet bbox definition
                bbox_[:, 2:] = bbox_[:, :2] + bbox_[:, 2:]
                outputs['bbox'].append(bbox_)
                outputs['lms'].append(det[:, 4:-1].reshape((det.shape[0], 5, 2)))
                outputs['idx'].extend([[i] * det.shape[0]])

        for key, value in outputs.items():
            value = np.concatenate(value)     
            outputs[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        
        return outputs

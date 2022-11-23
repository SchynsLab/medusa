import cv2
import numpy as np
from pathlib import Path

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
        conf, bbox, lms = [], [], []

        # Note to self: cv2 does not support batch prediction
        for i in range(b):

            _, out = self._model.detect(imgs[i, ...])

            if out is not None:                
                conf.append(out[:, -1])
                # Convert offset to true vertex positions to keep consistent
                # with retinanet bbox definition
                bbox_ = out[:, :4]
                bbox_[:, 2:] = bbox_[:, :2] + bbox_[:, 2:]
                bbox.append(bbox_)
                lms.append(out[:, 4:-1].reshape((out.shape[0], 5, 2)))
            else:
                conf.append(None)
                bbox.append(None)
                lms.append(None)

        return conf, bbox, lms

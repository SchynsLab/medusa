:py:mod:`medusa.detect.yunet`
=============================

.. py:module:: medusa.detect.yunet


Module Contents
---------------

.. py:class:: YunetDetector(det_threshold=0.5, nms_threshold=0.3, device=DEVICE, **kwargs)



   This detector is based on Yunet, a face detector based on YOLOv3 :cite:p:`facedetect-yu`.

   .. py:method:: forward(imgs)




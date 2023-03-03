:py:mod:`medusa.detect.scrfd`
=============================

.. py:module:: medusa.detect.scrfd

.. autoapi-nested-parse::

   SCRFD face detection model adapted from the insightface implementation. All
   numpy code has been converted to torch, speeding it up considerably (especially
   with GPU and large batch size). Please cite the corresponding paper `[1]` from
   the people at insightface if you use this implementation.

   Also, lease see the LICENSE file in the current directory for the
   license that is applicable to this implementation.



Module Contents
---------------

.. py:class:: SCRFDetector(det_size=(224, 224), det_threshold=0.5, nms_threshold=0.3, device=DEVICE)



   Face detection model based on the ``insightface`` package.

   :param name: Name of underlying insightface model
   :type name: str
   :param det_size: Size to which the input image(s) will be resized before passing it to the
                    detection model; should be a tuple with two of the same integers (indicating
                    a square image); higher values are more accurate but slower; default is
                    (224, 224)
   :type det_size: tuple
   :param det_threshold: Detection threshold (higher = more conservative); detections with confidence
                         values lower than ``det_threshold`` are discarded
   :type det_threshold: float
   :param nms_threshold: Non-maximum suppression threshold for predicted bounding boxes
   :type nms_threshold: float
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   .. rubric:: Examples

   To crop an image to be used for MICA reconstruction:

   >>> from medusa.data import get_example_frame
   >>> det_model = SCRFDetector()
   >>> img = get_example_frame()  # path to jpg image
   >>> det_results = det_model(img)
   >>> list(det_results.keys())
   ['img_idx', 'conf', 'lms', 'bbox', 'n_img']

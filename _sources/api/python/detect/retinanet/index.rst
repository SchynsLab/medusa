:py:mod:`medusa.detect.retinanet`
=================================

.. py:module:: medusa.detect.retinanet

.. autoapi-nested-parse::

   Face detection model adapted from the insightface implementation. By reimplementing
   it here, insightface does not have to be installed.

   Please see the LICENSE file in the current directory for the license that
   is applicable to this implementation.



Module Contents
---------------

.. py:class:: RetinanetDetector(det_threshold=0.5, nms_threshold=0.3, device=DEVICE)

   Bases: :py:obj:`medusa.detect.base.BaseDetectionModel`

   Face detection model based on the ``insightface`` package, as used
   by MICA (https://github.com/Zielon/MICA).

   :param name: Name of underlying insightface model
   :type name: str
   :param det_size: Image size for detection
   :type det_size: tuple
   :param target_size: Length 2 tuple with desired width/heigth of cropped image; should be (112, 112)
                       for MICA
   :type target_size: tuple
   :param det_thresh: Detection threshold (higher = more stringent)
   :type det_thresh: float
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   .. rubric:: Examples

   To crop an image to be used for MICA reconstruction:

   >>> from medusa.data import get_example_frame
   >>> crop_model = InsightfaceCropModel(device='cpu')
   >>> img = get_example_frame()  # path to jpg image
   >>> crop_img = crop_model(img)
   >>> crop_img.shape
   torch.Size([1, 3, 112, 112])

   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __call__(imgs, max_num=0)




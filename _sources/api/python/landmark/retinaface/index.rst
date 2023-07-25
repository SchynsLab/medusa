:py:mod:`medusa.landmark.retinaface`
====================================

.. py:module:: medusa.landmark.retinaface


Module Contents
---------------

.. py:class:: RetinafaceLandmarkModel(model_name='2d106det', detector=SCRFDetector, device=DEVICE)



   Landmark detection model based on Insightface's Retinaface model.

   :param model_name: Name of the landmark model from Insightface that should be used; options are
                      '2d106det' (106 landmarks, 2D) or '1k3d68' (68 landmarks, 3D)
   :type model_name: str
   :param detector: Which detector to use; options are ``SCRFDetector`` or ``YunetDetector``
   :type detector: BaseDetector
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   .. py:method:: forward(imgs)

      Runs the landmark model on a set of images.

      :param imgs: Either a list of images, a path to a directory containing images, or an
                   already loaded torch.tensor of shape N (batch) x C x H x W
      :type imgs: list, str, Path, torch.tensor

      :returns: **out_lms** -- Dictionary with the following keys: 'lms' (landmarks), 'conf' (confidence),
                'img_idx' (image index), 'bbox' (bounding box)
      :rtype: dict




:py:mod:`medusa.recognize.retinaface`
=====================================

.. py:module:: medusa.recognize.retinaface

.. autoapi-nested-parse::

   A face recognition model based on Insightface's Retinaface model, but implemented
   in PyTorch (but parts of it run with ONNX model), so can be fully run on GPU (no numpy
   necessary).



Module Contents
---------------

.. py:class:: RetinafaceRecognitionModel(model_path=None, device=DEVICE)



   Face recognition model based on Insightface's Retinaface model (trained using
   partial FC).

   :param model_path: Path to the ONNX model file; if None, will use the default model
   :type model_path: str, Path
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   .. py:method:: forward(imgs)

      Runs the recognition model on a set of images.

      :param imgs: Either a list of images, a path to a directory containing images, or an
                   already loaded torch.tensor of shape N (batch) x C x H x W
      :type imgs: list, str, Path, torch.tensor

      :returns: **X_emb** -- Face embeddings of shape N x 512 (where N is the number of detections, not
                necessarily the number of input images)
      :rtype: torch.tensor



.. py:class:: RetinafaceGenderAgeModel(model_path=None, device=DEVICE)



   Gender and age prediction model based on Insightface's model.

   :param model_path: Path to the ONNX model file; if None, will use the default model
   :type model_path: str, Path
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   .. py:method:: forward(imgs)

      Runs the gender+age prediction model on a set of images.

      :param imgs: A torch.tensor of shape N (batch) x C x H x W
      :type imgs: torch.tensor

      :returns: **attr** -- Dictionary with the following keys: 'gender' (0 = female, 1 = male),
                'age' (age in years), both torch.tensors with the first dimension being the
                number of faces detected in the input images
      :rtype: dict




:py:mod:`medusa.crop.bbox_crop`
===============================

.. py:module:: medusa.crop.bbox_crop

.. autoapi-nested-parse::

   Module with a "crop model" which crops an image by creating a bounding box
   based on a set of existing (2D) landmarks.

   Based on the implementation in DECA (see
   ../recon/flame/deca/license.md).



Module Contents
---------------

.. py:class:: BboxCropModel(lms_model_name='2d106det', output_size=(224, 224), device=DEVICE)



   A model that crops an image by creating a bounding box based on a set of
   face landmarks; based on the implementation from DECA.

   :param name: Name of the landmark model from Insightface that should be used; options are
                '2d106det' (106 landmarks) or '1k3d68' (68 landmarks)
   :type name: str
   :param output_size: Desired size of the cropped image
   :type output_size: tuple[int]
   :param detector: A Medusa-based detector
   :type detector: BaseDetector
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   .. py:method:: forward(imgs)

      Crops images to the desired size.

      :param imgs: A path to an image, or a tuple/list of them, or already loaded images
                   as a torch.tensor or numpy array
      :type imgs: str, Path, tuple, list, array_like, torch.tensor

      :returns: **out_crop** -- Dictionary with cropping outputs; includes the keys "imgs_crop" (cropped
                images) and "crop_mat" (3x3 crop matrices)
      :rtype: dict



.. py:class:: InsightfaceBboxCropModel(output_size=(192, 192), scale_factor=1.5, device=DEVICE)



   Crop model based on a (scaled and) square bounding box as implemented by
   insightface.

   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   .. py:method:: forward(imgs)

      Runs the crop model on a set of images.

      :param imgs: A torch.tensor of shape N (batch) x C x H x W
      :type imgs: torch.tensor

      :returns: **out_crop** -- Dictionary with cropping outputs; includes the keys "imgs_crop" (cropped
                images) and "crop_mat" (3x3 crop matrices)
      :rtype: dict




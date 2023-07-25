:py:mod:`medusa.crop.align_crop`
================================

.. py:module:: medusa.crop.align_crop

.. autoapi-nested-parse::

   Module with an implementation of a "crop model" that aligns an image to a
   template based on a set of landmarks (based on an implementation from
   Insightface).



Module Contents
---------------

.. py:data:: TEMPLATE

   The 5-landmark template used by Insightface (e.g. in their arcface implementation).
   The coordinates are relative to an image of size 112 x 112.

.. py:class:: AlignCropModel(output_size=(112, 112), template=TEMPLATE, detector=SCRFDetector, device=DEVICE, **kwargs)



   Cropping model based on functionality from the ``insightface`` package,
   as used by MICA (https://github.com/Zielon/MICA).

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

   >>> from medusa.data import get_example_image
   >>> crop_model = AlignCropModel()
   >>> img = get_example_image()  # path to jpg image
   >>> out = crop_model(img)

   .. py:method:: forward(imgs)

      Aligns and crops images to the desired size.

      :param imgs: A path to an image, or a tuple/list of them, or already loaded images
                   as a torch.tensor or numpy array
      :type imgs: str, Path, tuple, list, array_like, torch.tensor

      :returns: **out_crop** -- Dictionary with cropping outputs; includes the keys "imgs_crop" (cropped
                images) and "crop_mat" (3x3 crop matrices)
      :rtype: dict




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

.. py:class:: BboxCropModel(name='2d106det', output_size=(224, 224), detector=SCRFDetector, device=DEVICE, **kwargs)



   A model that crops an image by creating a bounding box based on a set of
   face landmarks.

   :param name: Name of the landmark model from Insightface that should be used; options are
                '2d106det' (106 landmarks) or '1k3d68' (68 landmarks)
   :type name: str
   :param output_size: Desired size of the cropped image
   :type output_size: tuple[int]
   :param detector: A Medusa-based detector
   :type detector: BaseDetector
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str



:py:mod:`medusa.recon.fan`
==========================

.. py:module:: medusa.recon.fan

.. autoapi-nested-parse::

   Module with functionality to use the FAN-3D model.

   This module contains a reconstruction model based on the ``face_alignment`` package
   by `Adrian Bulat <https://www.adrianbulat.com/face-alignment>`_ [1]_. It is used both
   as a reconstruction model as well as a way to estimate a bounding box as expected by
   the EMOCA model (which uses the bounding box to crop the original image).

   .. [1] Bulat, A., & Tzimiropoulos, G. (2017). How far are we from solving the 2d & 3d
          face alignment problem?(and a dataset of 230,000 3d facial landmarks).
          In *Proceedings of the IEEE International Conference on Computer Vision*
          (pp. 1021-1030).



Module Contents
---------------

.. py:data:: logger
   

   

.. py:class:: FAN(device='cpu', target_size=224, face_detector='sfd', lm_type='2D', use_prev_fan_bbox=False, use_prev_bbox=False)

   A wrapper around the FAN-3D landmark prediction model.

   .. attribute:: model

      The actual face alignment model

      :type: face_alignment.FaceAlignment

   .. py:method:: prepare_for_emoca(self, image)

      Runs all steps of the cropping / preprocessing pipeline
      necessary for use with DECA/EMOCA.

      :param image: Either a string or ``pathlib.Path`` object to an image or a numpy array
                    (width x height x 3) representing the already loaded RGB image
      :type image: str, Path, np.ndarray

      :returns: **img** -- The preprocessed (normalized) and cropped image as a ``torch.Tensor``
                of shape (1, 3, 224, 224), as EMOCA expects (the 1 is the batch size)
      :rtype: torch.Tensor

      .. rubric:: Examples

      To preprocess (which includes cropping) an image:

      >>> from medusa.data import get_example_frame
      >>> img = get_example_frame()
      >>> model = FAN(lm_type='2D')
      >>> cropped_img = model.prepare_for_emoca(img)
      >>> tuple(cropped_img.size())  #
      (1, 3, 224, 224)


   .. py:method:: __call__(self, image=None)

      Estimates (2D) landmarks (vertices) on the face.

      :param image: Path (str or pathlib Path) pointing to image file or 3D numpy array
                    (with np.uint8 values) representing a RGB image
      :type image: str, Path, or numpy array

      :raises ValueError : if `image` is `None` *and* self.img_orig is `None`:


   .. py:method:: viz_qc(self, f_out=None, return_rgba=False)

      Visualizes the inferred 2D landmarks & bounding box, as well as the final
      cropped image.

      f_out : str, Path
          Path to save viz to
      return_rgba : bool
          Whether to return a numpy image with the raw pixel RGBA intensities
          (True) or not (False; return nothing)




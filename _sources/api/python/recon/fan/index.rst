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

   :param device: Device to use, either 'cpu' or 'cuda' (for GPU)
   :type device: str
   :param target_size: Size to crop the image to, assuming a square crop; default (224)
                       corresponds to image size that EMOCA expects (ignored if not calling
                       ``prepare_for_emoca``)
   :type target_size: int
   :param face_detector: Face detector algorithm to use (default: 'sfd', as used in EMOCA)
   :type face_detector: str
   :param lm_type: Either '2D' (using 2D landmarks, necessary when using it for EMOCA) or
                   '3D' (using 3D landmarks), necessary when using it as a reconstruction
                   model
   :type lm_type: str
   :param use_prev_fan_bbox: Whether to use the previous bbox from FAN to do an initial crop (True)
                             or whether to run the FAN face detection algorithm again (False)
   :type use_prev_fan_bbox: bool
   :param use_prev_bbox: Whether to use the previous DECA-style bbox (True) or whether to
                         run FAN again to estimate landmarks from which to create a new
                         bbox (False); this should only be used when there is very little
                         rigid motion of the face!
   :type use_prev_bbox: bool

   .. attribute:: model

      The actual face alignment model

      :type: face_alignment.FaceAlignment

   .. rubric:: Examples

   To create a FAN based reconstruction model:

   >>> recon_model = FAN(lm_type='3D')

   To create a FAN-2D model for cropping images (as expected by EMOCA):

   >>> recon_model = FAN(lm_type='2D')

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

      Estimates landmarks (vertices) on the face.

      :param image: Either a string or ``pathlib.Path`` object to an image or a numpy array
                    (width x height x 3) representing the already loaded RGB image
      :type image: str, Path, np.ndarray

      :returns: **out** -- A dictionary with one key: ``"v"``, the reconstructed vertices (68 in
                total) with 2 (if using ``lm_type='2D'``) or 3 (if using ``lm_type='3D'``)
                coordinates
      :rtype: dict

      .. rubric:: Examples

      To reconstruct an example, simply call the ``FAN`` object:

      >>> from medusa.data import get_example_frame
      >>> model = FAN(lm_type='3D')
      >>> img = get_example_frame()
      >>> out = model(img)  # reconstruct!
      >>> out['v'].shape    # vertices
      (68, 3)


   .. py:method:: viz_qc(self, f_out=None, return_rgba=False)

      Visualizes the inferred 3D landmarks & bounding box, as well as the final
      cropped image.

      :param f_out: Path to save viz to; if ``None``, returned as an RGBA image
      :type f_out: str, Path
      :param return_rgba: Whether to return a numpy image with the raw pixel RGBA intensities
                          (True) or not (False; return nothing)
      :type return_rgba: bool

      :returns: **img** -- The rendered image as a numpy array (if ``f_out`` is ``None``)
      :rtype: np.ndarray

      .. rubric:: Examples

      To visualize the landmark and (EMOCA-style) bounding box:

      >>> from medusa.data import get_example_frame
      >>> img = get_example_frame()
      >>> fan = FAN(lm_type='2D')
      >>> cropped_img = fan.prepare_for_emoca(img)
      >>> viz_img = fan.viz_qc(return_rgba=True)
      >>> viz_img.shape
      (480, 640, 4)




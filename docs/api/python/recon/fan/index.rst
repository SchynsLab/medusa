:py:mod:`gmfx.recon.fan`
========================

.. py:module:: gmfx.recon.fan


Module Contents
---------------

.. py:data:: logger
   

   

.. py:class:: FAN(device='cpu', target_size=224, face_detector='sfd', lm_type='2D', use_prev_fan_bbox=False, use_prev_bbox=False)

   FAN face detection and landmark estimation, as implemented by
   Bulat & Tzimiropoulos (2017, Arxiv), adapted for use with DECA
   by Yao Feng (https://github.com/YadiraF), and further modified by
   Lukas Snoek

   :param device: Device to use, either 'cpu' or 'cuda' (for GPU)
   :type device: str
   :param target_size: Size to crop the image to, assuming a square crop; default (224)
                       corresponds to image size that DECA expects
   :type target_size: int
   :param face_detector: Face detector algorithm to use (default: 'sfd', as used in DECA)
   :type face_detector: str
   :param lm_type: Either '2D' (using 2D landmarks) or '3D' (using 3D landmarks)
   :type lm_type: str
   :param use_prev_fan_bbox: Whether to use the previous bbox from FAN to do an initial crop (True)
                             or whether to run the FAN face detection algorithm again (False)
   :type use_prev_fan_bbox: bool
   :param use_prev_bbox: Whether to use the previous DECA-style bbox (True) or whether to
                         run FAN again to estimate landmarks from which to create a new
                         bbox (False); this should only be used when there is very little
                         rigid motion of the face!
   :type use_prev_bbox: bool

   .. py:method:: prepare_for_emoca(self, image)

      Runs all steps of the cropping / preprocessing pipeline
      necessary for use with DECA/EMOCA.


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




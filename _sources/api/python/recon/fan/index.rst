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
   

   

.. py:class:: FAN(device='cpu', use_prev_bbox=True, min_detection_threshold=0.5, **kwargs)

   Bases: :py:obj:`medusa.recon.base.BaseModel`

   A wrapper around the FAN-3D landmark prediction model.

   :param device: Device to use, either 'cpu' or 'cuda' (for GPU)
   :type device: str
   :param use_prev_bbox: Whether to use the previous bbox from FAN to do an initial crop (True)
                         or whether to run the FAN face detection algorithm again (False)
   :type use_prev_bbox: bool

   .. attribute:: model

      The actual face alignment model

      :type: face_alignment.FaceAlignment

   .. rubric:: Examples

   To create a FAN based reconstruction model:

   >>> recon_model = FAN(device='cpu')

   .. py:method:: get_faces()

      FAN only returns landmarks, not a full mesh.


   .. py:method:: __call__(image=None)

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
      >>> model = FAN(device='cpu')
      >>> img = get_example_frame()
      >>> out = model(img)  # reconstruct!
      >>> out['v'].shape    # vertices
      (68, 3)




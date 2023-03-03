:py:mod:`medusa.recon.mpipe.mpipe`
==================================

.. py:module:: medusa.recon.mpipe.mpipe

.. autoapi-nested-parse::

   Module with a wrapper around a Mediapipe face mesh model [1]_ that can be
   used in Medusa.

   .. [1] Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019).
          Real-time facial surface geometry from monocular video on mobile GPUs.
          *arXiv preprint arXiv:1907.06724*



Module Contents
---------------

.. py:class:: Mediapipe(static_image_mode=False, det_threshold=0.1, device=DEVICE, **kwargs)



   A Mediapipe face mesh reconstruction model.

   :param static_image_mode: Whether to expect a sequence of related images (like in a video)
   :type static_image_mode: bool
   :param det_threshold: Minimum detection threshold (default set to 0.1 because lots of false negatives)
   :type det_threshold: float
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str
   :param \*\*kwargs: Extra keyword arguments to be passed to the initialization of FaceMesh
   :type \*\*kwargs: dict

   .. attribute:: model

      The actual Mediapipe model object

      :type: mediapipe.solutions.face_mesh.FaceMesh

   .. py:method:: get_tris()

      Returns the triangles associated with the mediapipe mesh.


   .. py:method:: get_cam_mat()

      Returns a default camera matrix.


   .. py:method:: close()

      Closes context manager.

      Ideally, after you're doing with reconstructing each frame of
      the video, you call this method to close the manually opened
      context (but shouldn't matter much if you only instantiate a
      single model).

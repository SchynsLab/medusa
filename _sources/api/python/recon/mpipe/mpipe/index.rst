:py:mod:`medusa.recon.mpipe.mpipe`
==================================

.. py:module:: medusa.recon.mpipe.mpipe

.. autoapi-nested-parse::

   Module with a wrapper around a Mediapipe face mesh model [1]_ that can be
   used in Medusa.

   .. [1] Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019).
      Real-time facial surface geometry from monocular video on mobile GPUs. *arXiv
      preprint arXiv:1907.06724*



Module Contents
---------------

.. py:class:: Mediapipe(static_image_mode=True, **kwargs)

   A Mediapipe face mesh reconstruction model.

   :param static_image_mode: Whether to expect a sequence of related images
                             (like in a video)
   :type static_image_mode: bool
   :param kwargs: Extra keyword arguments to be passed to
                  the initialization of FaceMesh
   :type kwargs: dict

   .. attribute:: model

      The actual Mediapipe model object

      :type: mediapipe.solutions.face_mesh.FaceMesh

   .. py:method:: get_faces(self)


   .. py:method:: __call__(self, image)

      Performs reconstruction of the face as a list of landmarks (vertices).

      :param image: A 3D (w x h x 3) numpy array representing a RGB image
      :type image: np.ndarray

      :returns: **out** -- A dictionary with two keys: ``"v"``, the reconstructed vertices (468 in
                total) and ``"mat"``, a 4x4 Numpy array representing the local-to-world
                matrix
      :rtype: dict

      .. rubric:: Notes

      This implementation returns 468 vertices instead of the original 478, because
      the last 10 vertices (representing the irises) are not present in the canonical
      model.

      .. rubric:: Examples

      To reconstruct an example, simply call the ``Mediapipe`` object:

      >>> from medusa.data import get_example_frame
      >>> model = Mediapipe()
      >>> img = get_example_frame()
      >>> out = model(img)  # reconstruct!
      >>> out['v'].shape    # vertices
      (468, 3)
      >>> out['mat'].shape  # local-to-world matrix
      (4, 4)


   .. py:method:: close(self)

      Closes context manager.

      Ideally, after you're doing with reconstructing each frame of the video,
      you call this method to close the manually opened context (but shouldn't
      matter much if you only instantiate a single model).




:py:mod:`medusa.render`
=======================

.. py:module:: medusa.render

.. autoapi-nested-parse::

   Module with functionality to render 4D face mesh data.

   The ``Renderer`` class is a high-level wrapper around functionality from the
   excellent `pyrender <https://pyrender.readthedocs.io>`_ package [1]_.

   .. [1] Matl, Matthew. *pyrender* [computer software]. https://github.com/mmatl/pyrender



Module Contents
---------------

.. py:class:: Renderer(viewport, camera_type='orthographic', smooth=True, wireframe=False, cam_mat=None)

   A high-level wrapper around a pyrender-based renderer.

   :param viewport: Desired output image size (width, height), in pixels; should match
                    the original image (before cropping) that was reconstructed
   :type viewport: tuple[int]
   :param camera_type: Either 'orthographic' (for Flame-based reconstructions) or
                       'intrinsic' (for mediapipe reconstruction)
   :type camera_type: str
   :param smooth: Whether to render a smooth mesh (by normal interpolation) or not
   :type smooth: bool
   :param wireframe: Whether to render a wireframe instead of a surface
   :type wireframe: bool
   :param zoom_out: How much to translate the camera into the positive z direction
                    (necessary for Flame-based reconstructions)
   :type zoom_out: int/float

   .. py:method:: __call__(self, v, f)

      Performs the actual rendering.

      :param v: A 2D array with vertices of shape V (nr of vertices) x 3
                (X, Y, Z)
      :type v: np.ndarray
      :param f: A 2D array with 'faces' (triangles) of shape F (nr of faces) x 3
                (nr of vertices); should be integers
      :type f: np.ndarray

      :returns: **img** -- A 3D array (with np.uint8 integers) of shape `viewport[0]` x
                `viewport[1]` x 3 (RGB)
      :rtype: np.ndarray


   .. py:method:: alpha_blend(self, img, background, face_alpha=None)

      Simple alpha blend of a rendered image and
      a background. The image (`img`) is assumed to be
      an RGBA image and the background (`background`) is
      assumed to be a RGB image. The alpha channel of the image
      is used to blend them together. The optional `threshold`
      parameter can be used to impose a sharp cutoff.

      :param img: A 3D numpy array of shape height x width x 4 (RGBA)
      :type img: np.ndarray
      :param background: A 3D numpy array of shape height x width x 3 (RGB)
      :type background: np.ndarray


   .. py:method:: close(self)

      Closes the OffScreenRenderer object.




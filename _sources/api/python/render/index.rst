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

.. py:class:: Renderer(viewport, camera_type='orthographic', smooth=True, wireframe=False, cam_mat=None, focal_length=None)

   A high-level wrapper around a pyrender-based renderer.

   :param viewport: Desired output image size (width, height), in pixels; should match
                    the original image (before cropping) that was reconstructed
   :type viewport: tuple[int]
   :param camera_type: Either 'orthographic' (for Flame-based reconstructions) or
                       'intrinsic' (for mediapipe reconstruction, i.e., a perspective camera)
   :type camera_type: str
   :param smooth: Whether to render a smooth mesh (by normal interpolation) or not
   :type smooth: bool
   :param wireframe: Whether to render a wireframe instead of a surface
   :type wireframe: bool

   .. py:method:: __call__(v, f, overlay=None, cmap_name='bwr', is_colors=False)

      Performs the actual rendering.

      :param v: A 2D array with vertices of shape V (nr of vertices) x 3
                (X, Y, Z)
      :type v: np.ndarray
      :param f: A 2D array with 'faces' (triangles) of shape F (nr of faces) x 3
                (nr of vertices); should be integers
      :type f: np.ndarray
      :param overlay: A 1D array with overlay values (numpy floats between 0 and 1), one for each
                      vertex or face
      :type overlay: np.ndarray
      :param cmap_name: Name of (matplotlib) colormap; only relevant if ``overlay`` is not ``None``
      :type cmap_name: str
      :param is_colors: If ``True``, then ``overlay`` is a V (of F) x 4 (RGBA) array; if ``False``,
                        ``overlay`` is assumed to be a 1D array with floats betwee 0 and 1
      :type is_colors: bool

      :returns: **img** -- A 3D array (with np.uint8 integers) of shape ``viewport[0]`` x
                ``viewport[1]`` x 3 (RGB)
      :rtype: np.ndarray


   .. py:method:: alpha_blend(img, background, face_alpha=None)

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


   .. py:method:: close()

      Closes the OffScreenRenderer object.




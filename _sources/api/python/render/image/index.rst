:py:mod:`medusa.render.image`
=============================

.. py:module:: medusa.render.image

.. autoapi-nested-parse::

   Module with a renderer class based on ``pytorch3d``.



Module Contents
---------------

.. py:class:: PytorchRenderer(viewport, cam_mat, cam_type, shading='flat', lights=None, background=(0, 0, 0), device=DEVICE)



   A pytorch3d-based renderer.

   :param viewport: Desired output image size (width, height), in pixels; should match
                    the original image (before cropping) that was reconstructed
   :type viewport: tuple[int]
   :param cam_mat: A camera matrix to set the position/angle of the camera
   :type cam_mat: torch.tensor
   :param cam_type: Either 'orthographic' (for Flame-based reconstructions) or
                    'perpective' (for mediapipe reconstructions)
   :type cam_type: str
   :param shading: Type of shading ('flat', 'smooth')
   :type shading: str
   :param wireframe_opts: Dictionary with extra options for wireframe rendering (options: 'width', 'color')
   :type wireframe_opts: None, dict
   :param device: Device to store the image on ('cuda' or 'cpu')
   :type device: str

   .. py:method:: draw_landmarks(img, lms, radius=3, colors='green')


   .. py:method:: alpha_blend(img, background, face_alpha=None)

      Simple alpha blend of a rendered image and a background. The image
      (`img`) is assumed to be an RGBA image and the background
      (`background`) is assumed to be a RGB image. The alpha channel of the
      image is used to blend them together. The optional `threshold`
      parameter can be used to impose a sharp cutoff.

      :param img: A 3D or 4D tensor of shape (batch size) x height x width x 4 (RGBA)
      :type img: torch.tensor
      :param background: A 3D or 4D tensor shape height x width x 3 (RGB[A])
      :type background: np.ndarray

      :returns: **img** -- A blended image
      :rtype: torch.tensor


   .. py:method:: save_image(f_out, img)
      :staticmethod:

      Saves a single image (using ``PIL``) to disk.

      :param f_out: Path where the image should be saved
      :type f_out: str, Path


   .. py:method:: forward(v, tris, overlay=None, single_image=True)

      Performs the actual rendering for a given (batch of) mesh(es).

      :param v: A 3D (batch size x vertices x 3) tensor with vertices
      :type v: torch.tensor
      :param tris: A 3D (batch size x vertices x 3) tensor with triangles
      :type tris: torch.tensor
      :param overlay: A tensor with shape (batch size x vertices) with vertex colors
      :type overlay: torch.tensor
      :param single_image: Whether a single image with (potentially) multiple faces should be
                           renderer (True) or multiple images with a single face should be renderered
                           (False)
      :type single_image: bool

      :returns: **img** -- A 4D tensor with uint8 values of shape batch size x h x w x 3 (RGB), where
                h and w are defined in the viewport
      :rtype: torch.tensor




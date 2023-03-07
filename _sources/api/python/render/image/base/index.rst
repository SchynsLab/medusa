:py:mod:`medusa.render.image.base`
==================================

.. py:module:: medusa.render.image.base

.. autoapi-nested-parse::

   Module with a renderer base class.



Module Contents
---------------

.. py:class:: BaseRenderer



   A base class for the renderers in Medusa.

   .. py:method:: close()
      :abstractmethod:

      Closes the currently used renderer.


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


   .. py:method:: load_image(f_in, device=None)
      :staticmethod:

      Utility function to read a single image to disk (using ``PIL``).

      :param f_in: Path of file
      :type f_in: str, Path
      :param device: If ``None``, the image is returned as a numpy array; if 'cuda' or 'cpu',
                     the image is returned as a torch tensor
      :type device: None, str




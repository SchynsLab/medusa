:py:mod:`medusa.render.image.pytorch3d`
=======================================

.. py:module:: medusa.render.image.pytorch3d

.. autoapi-nested-parse::

   Module with a renderer class based on ``pytorch3d``.



Module Contents
---------------

.. py:class:: PytorchRenderer(viewport, cam_mat, cam_type, shading='flat', lights=None, device=DEVICE)



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

   .. py:method:: normal_map(fragments=None, v=None, tris=None)


   .. py:method:: add_sh_light(img, fragments, sh_coeff)


   .. py:method:: close()

      Closes the currently used renderer.




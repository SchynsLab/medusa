:py:mod:`medusa.render.lights`
==============================

.. py:module:: medusa.render.lights

.. autoapi-nested-parse::

   Adapted from https://github.com/pomelyu/SHLight_pytorch, by Chien Chin-yu.



Module Contents
---------------

.. py:class:: SphericalHarmonicsLights(sh_params, ambient_color=(0.5, 0.5, 0.5), device=DEVICE)



   An implementation of spherical harmonics lighting, adapted to work with
   the SH parameters returned by DECA/EMOCA.

   :param sh_params: Tensor of shape (B, 9, 3) containing SH parameters
   :type sh_params: torch.Tensor
   :param ambient_color: Color of ambient light
   :type ambient_color: tuple
   :param device: Device to use (e.g., "cuda", "cpu")
   :type device: str

   .. py:method:: clone()

      Clones the object.


   .. py:method:: diffuse(normals, points) -> torch.Tensor

      Computes diffuse lighting.


   .. py:method:: specular(normals, points, camera_position, shininess) -> torch.Tensor

      Computes specular lighting.




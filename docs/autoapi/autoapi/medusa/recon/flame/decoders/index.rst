:py:mod:`medusa.recon.flame.decoders`
=====================================

.. py:module:: medusa.recon.flame.decoders

.. autoapi-nested-parse::

   Decoder-modules for FLAME-based reconstruction models.

   See ./deca/license.md for conditions for use.



Module Contents
---------------

.. py:class:: FLAME(model_path, n_shape, n_exp)



   Generates a FLAME-based based on latent parameters.

   .. py:method:: forward(shape_params=None, expression_params=None, pose_params=None)

      Input:
          shape_params: N X number of shape parameters
          expression_params: N X number of expression parameters
          pose_params: N X number of pose parameters (6)
      return:d
          vertices: N X V X 3
          landmarks: N X number of landmarks X 3



.. py:class:: FLAMETex(model_path=None, n_tex=50)



   FLAME texture:

   https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
   FLAME texture converted from BFM:
   https://github.com/TimoBolkart/BFM_to_FLAME

   .. py:method:: forward(texcode)

      texcode: [batchsize, n_tex]
      texture: [bz, 3, 256, 256], range: 0-1



.. py:function:: to_tensor(array, dtype=torch.float32)


.. py:function:: to_np(array, dtype=np.float32)

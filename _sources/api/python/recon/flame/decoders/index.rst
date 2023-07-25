:py:mod:`medusa.recon.flame.decoders`
=====================================

.. py:module:: medusa.recon.flame.decoders

.. autoapi-nested-parse::

   Decoder-modules for FLAME-based reconstruction models.

   See ./deca/license.md for conditions for use.



Module Contents
---------------

.. py:class:: FlameShape(n_shape=300, n_expr=100, parameters=None, device=DEVICE, **init_parameters)



   Generates a FLAME-based mesh (shape only) from 3DMM parameters.


   .. py:method:: get_full_pose()

      Returns the full pose vector.


   .. py:method:: forward(batch_size=None, **inputs)

      Input:
          shape_params: N X number of shape parameters
          expression_params: N X number of expression parameters
          pose_params: N X number of pose parameters (6)
      return:d
          vertices: N X V X 3
          landmarks: N X number of landmarks X 3



.. py:class:: FlameLandmark(lm_type='68', lm_dim='2d', device=DEVICE)



   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:method:: forward(v, poses)



.. py:class:: FlameTex(model_path=None, n_tex=50)



   FLAME texture:

   https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
   FLAME texture converted from BFM:
   https://github.com/TimoBolkart/BFM_to_FLAME

   .. py:method:: forward(texcode)

      texcode: [batchsize, n_tex]
      texture: [bz, 3, 256, 256], range: 0-1




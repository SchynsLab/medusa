:py:mod:`medusa.recon.emoca.models.decoders`
============================================

.. py:module:: medusa.recon.emoca.models.decoders


Module Contents
---------------

.. py:class:: Generator(latent_dim=100, out_channels=1, out_scale=0.01, sample_mode='bilinear')

   Bases: :py:obj:`torch.nn.Module`

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

   .. py:method:: forward(self, noise)



.. py:class:: FLAME(config, n_shape, n_exp)

   Bases: :py:obj:`torch.nn.Module`

   borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
   Given flame parameters this class generates a differentiable FLAME function
   which outputs the a mesh and 2D/3D facial landmarks

   .. py:method:: forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None)

      Input:
          shape_params: N X number of shape parameters
          expression_params: N X number of expression parameters
          pose_params: N X number of pose parameters (6)
      return:d
          vertices: N X V X 3
          landmarks: N X number of landmarks X 3



.. py:class:: FLAMETex(config, n_tex)

   Bases: :py:obj:`torch.nn.Module`

   FLAME texture:
   https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
   FLAME texture converted from BFM:
   https://github.com/TimoBolkart/BFM_to_FLAME

   .. py:method:: forward(self, texcode)

      texcode: [batchsize, n_tex]
      texture: [bz, 3, 256, 256], range: 0-1



.. py:function:: to_tensor(array, dtype=torch.float32)


.. py:function:: to_np(array, dtype=np.float32)


.. py:class:: Struct(**kwargs)

   Bases: :py:obj:`object`



:py:mod:`medusa.recon.emoca.emoca`
==================================

.. py:module:: medusa.recon.emoca.emoca


Module Contents
---------------

.. py:data:: benchmark
   :annotation: = True

   

.. py:class:: EMOCA(img_size, cfg=None, device='cuda')

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

   .. py:method:: encode(self, image)

      "Encodes" the image into FLAME parameters, i.e., predict FLAME
      parameters for the given image. Note that, at the moment, it only
      works for a single image, not a batch of images.

      :param image: A Tensor with shape 1 (batch size) x 3 (color ch.) x 244 (w) x 244 (h)
      :type image: torch.Tensor

      :returns: **enc_dict** -- A dictionary with all encoded parameters and some extra data needed
                for the decoding stage.
      :rtype: dict


   .. py:method:: decode(self, enc_dict)

      Decodes the face attributes (vertices, landmarks, texture, detail map)
      from the encoded parameters.

      :param orig_size: Tuple containing the original image size (height, width), i.e.,
                        before cropping; needed to transform and render the mesh in the
                        original image space
      :type orig_size: tuple

      :returns: **dec_dict** -- A dictionary with the results from the decoding stage
      :rtype: dict

      :raises ValueError: If `tform` parameter is not `None` and `orig_size` is `None`. In other
          words, if `tform` is supplied, `orig_size` should be supplied as well


   .. py:method:: forward(self, img)




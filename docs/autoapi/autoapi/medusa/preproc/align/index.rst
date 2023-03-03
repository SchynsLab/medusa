:py:mod:`medusa.preproc.align`
==============================

.. py:module:: medusa.preproc.align


Module Contents
---------------

.. py:function:: estimate_alignment(v, topo, target=None, estimate_scale=False, device=DEVICE)

   Aligment of a temporal series of 3D meshes to a target (which should
   have the same topology).

   :param v: A float tensor with vertices of shape B (batch size) x V (vertices) x 3
   :type v: torch.tensor
   :param topo: Topology corresponding to ``v``
   :type topo: str
   :param target: Target to use for alignment; if ``None`` (default), a default template will be
                  used
   :type target: torch.tensor
   :param estimate_scale: Whether the alignment may also involve scaling
   :type estimate_scale: bool
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   :returns: **mat** -- A float tensor with affine matrices of shape B (batch size) x 4 x 4
   :rtype: torch.tensor

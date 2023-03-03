:py:mod:`medusa.geometry`
=========================

.. py:module:: medusa.geometry

.. autoapi-nested-parse::

   Module with geometry-related functionality.

   For now only contains functions to compute vertex and triangle normals.



Module Contents
---------------

.. py:function:: compute_tri_normals(v, tris, normalize=True)

   Computes triangle (surface/face) normals.

   :param v: A float tensor with vertices of shape B (batch size) x V (vertices) x 3
   :type v: torch.tensor
   :param tris: A long tensor with indices of shape T (triangles) x 3 (vertices per triangle)
   :type tris: torch.tensor
   :param normalize: Whether to normalize the normals (usually, you want to do this, but included
                     here so it can be reused when computing vertex normals, which uses
                     unnormalized triangle normals)
   :type normalize: bool

   :returns: **fn** -- A float tensor with triangle normals of shape B (batch size) x T (triangles) x 3
   :rtype: torch.tensor


.. py:function:: compute_vertex_normals(v, tris)

   Computes vertex normals in a vectorized way, based on the ``pytorch3d``
   implementation.

   :param v: A float tensor with vertices of shape B (batch size) x V (vertices) x 3
   :type v: torch.tensor
   :param tris: A long tensor with indices of shape T (triangles) x 3 (vertices per triangle)
   :type tris: torch.tensor

   :returns: **vn** -- A float tensor with vertex normals of shape B (batch size) x V (vertices) x 3
   :rtype: torch.tensor


.. py:function:: apply_vertex_mask(name, **attrs)

   Applies a vertex mask to a tensor of vertices.

   :param v: A float tensor with vertices of shape B (batch size) x V (vertices) x 3
   :type v: torch.tensor
   :param name: Name of mask to apply
   :type name: str

   :returns: **v_masked** -- A float tensor with masked vertices of shape B (batch size) x V (vertices) x 3
   :rtype: torch.tensor

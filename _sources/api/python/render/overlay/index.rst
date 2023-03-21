:py:mod:`medusa.render.overlay`
===============================

.. py:module:: medusa.render.overlay


Module Contents
---------------

.. py:class:: Overlay(v, cmap='bwr', vmin=None, vmax=None, vcenter=None, dim='normals', v0=None, tris=None, norm=TwoSlopeNorm)

   Class for creating color "overlays" to be rendered as vertex colors of a mesh.

   :param v: The values to be mapped to colors
   :type v: torch.Tensor
   :param vmin: The minimum value of the colormap (default: minimum of v)
   :type vmin: float, optional
   :param vmax: The maximum value of the colormap (default: maximum of v)
   :type vmax: float, optional
   :param vcenter: The center value of the colormap (default: mean of v)
   :type vcenter: float, optional
   :param dim: If int (0, 1, 2), dimension to be visualized; if 'normals', the values are
               projected on the vertex normals
   :type dim: int or str
   :param v0: If dim='normals', v0 represents the mesh from which the normals are computed
   :type v0: torch.Tensor, optional
   :param tris: If dim='normals', tris represents the triangles from which the normals are computed
   :type tris: torch.Tensor, optional
   :param norm: Normalization class (default: `TwoSlopeNorm`)
   :type norm: matplotlib.colors.Normalize, optional

   .. py:method:: to_rgb()

      Returns the RGB colors for the overlay.

      :returns: **colors** -- RGB colors (N x V x 3)
      :rtype: torch.Tensor




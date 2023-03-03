:py:mod:`medusa.recon.flame.utils`
==================================

.. py:module:: medusa.recon.flame.utils


Module Contents
---------------

.. py:function:: face_vertices(v, f)


.. py:function:: vertex_normals(v, f)

   :param vertices: [batch size, number of vertices, 3]
   :param faces: [batch size, number of faces, 3]
   :return: [batch size, number of vertices, 3]


.. py:function:: upsample_mesh(v, normals, disp_map, dense_template)

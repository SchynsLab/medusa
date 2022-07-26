:py:mod:`medusa.transforms`
===========================

.. py:module:: medusa.transforms


Module Contents
---------------

.. py:function:: apply_perspective_projection(v, mat)

   " Applies a perspective projection of ``v`` into NDC space.

   :param v: A 2D (vertices x XYZ) array with vertex data
   :type v: np.ndarray
   :param mat: A 4x4 perspective projection matrix
   :type mat: np.ndarray


.. py:function:: embed_points_in_mesh(v, f, p)

   Embed points in an existing mesh by finding the face it is contained in and
   computing its barycentric coordinates. Works with either 2D or 3D data.

   :param v: Vertices of the existing mesh (a 2D vertices x [2 or 3] array)
   :type v: np.ndarray
   :param f: Faces (polygons) of the existing mesh (a 2D faces x [2 or 3] array)
   :type f: np.ndarray
   :param p: Points (vertices) to embed (a 2D vertices x [2 or 3] array)
   :type p: np.ndarray

   :returns: * **triangles** (*np.ndarray*) -- A 1D array with faces corresponding to the vertices of ``p``
             * **bcoords** (*np.ndarray*) -- A 2D array (vertices x 3) array with barycentric coordinates


.. py:function:: project_points_from_embedding(v, f, triangles, bcoords)

   Project points (vertices) from an existing embedding into a different space.

   :param v: Points (vertices) to project
   :type v: np.ndarray
   :param f: Faces of original mesh
   :type f: np.ndarray
   :param triangles:



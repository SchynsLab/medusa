:py:mod:`medusa.transforms`
===========================

.. py:module:: medusa.transforms


Module Contents
---------------

.. py:function:: create_viewport_matrix(nx, ny, device='cuda')

   Creates a viewport matrix that transforms vertices in NDC [-1, 1]
   space to viewport (screen) space. Based on a blogpost by Mauricio Poppe:
   https://www.mauriciopoppe.com/notes/computer-graphics/viewing/viewport-transform/
   except that I added the minus sign at [1, 1], which makes sure that the
   viewport (screen) space origin is in the top left.

   :param nx: Number of pixels in the x dimension (width)
   :type nx: int
   :param ny: Number of pixels in the y dimension (height)
   :type ny: int

   :returns: **mat** -- A 4x4 numpy array representing the viewport transform
   :rtype: np.ndarray


.. py:function:: create_ortho_matrix(nx, ny, znear=0.05, zfar=100.0, device='cuda')

   Creates an orthographic projection matrix, as
   used by EMOCA/DECA. Based on the pyrender implementaiton.
   Assumes an xmag and ymag of 1.

   :param nx: Number of pixels in the x-dimension (width)
   :type nx: int
   :param ny: Number of pixels in the y-dimension (height)
   :type ny: int
   :param znear: Near clipping plane distance (from eye/camera)
   :type znear: float
   :param zfar: Far clipping plane distance (from eye/camera)
   :type zfar: float

   :returns: **mat** -- A 4x4 affine matrix
   :rtype: np.ndarray


.. py:function:: crop_matrix_to_3d(mat_33)

   Transforms a 3x3 matrix used for cropping (on 2D coordinates)
   into a 4x4 matrix that can be used to transform 3D vertices.
   It assumes that there is no rotation element.

   :param mat_33: A 3x3 affine matrix
   :type mat_33: np.ndarray

   :returns: **mat_44** -- A 4x4 affine matrix
   :rtype: np.ndarray


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

   :param v: Points (vertices) to project (:math:`N  imes 3`)
   :type v: np.ndarray
   :param f: Faces of original mesh
   :type f: np.ndarray
   :param triangles:
   :type triangles: np.ndarray


.. py:function:: estimate_similarity_transform(src, dst, estimate_scale=True)

   Estimate a similarity transformation matrix for two batches
   of points with N observations and M dimensions; reimplementation
   of the ``_umeyama`` function of the scikit-image package.

   :param src: A tensor with shape batch_size x N x M
   :type src: torch.Tensor
   :param dst: A tensor with shape batch_size x N x M
   :type dst: torch.Tensor
   :param estimate_scale: Whether to also estimate a scale parameter
   :type estimate_scale: bool

   :raises ValueError: When N (number of points) < M (number of dimensions)



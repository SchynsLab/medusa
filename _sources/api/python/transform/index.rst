:py:mod:`medusa.transform`
==========================

.. py:module:: medusa.transform


Module Contents
---------------

.. py:function:: create_viewport_matrix(nx, ny)

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


.. py:function:: create_ortho_matrix(nx, ny, znear=0.05, zfar=100.0)

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



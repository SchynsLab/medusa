import numpy as np


def create_viewport_matrix(nx, ny):
    """Creates a viewport matrix that transforms vertices in NDC [-1, 1]
    space to viewport (screen) space. Based on a blogpost by Mauricio Poppe:
    https://www.mauriciopoppe.com/notes/computer-graphics/viewing/viewport-transform/
    except that I added the minus sign at [1, 1], which makes sure that the
    viewport (screen) space origin is in the top left.

    Parameters
    ----------
    nx : int
        Number of pixels in the x dimension (width)
    ny : int
        Number of pixels in the y dimension (height)

    Returns
    -------
    mat : np.ndarray
        A 4x4 numpy array representing the viewport transform
    """

    mat = np.array(
        [
            [nx / 2, 0, 0, (nx - 1) / 2],
            [0, -ny / 2, 0, (ny - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    return mat


def create_ortho_matrix(nx, ny, znear=0.05, zfar=100.0):
    """Creates an orthographic projection matrix, as
    used by EMOCA/DECA. Based on the pyrender implementaiton.
    Assumes an xmag and ymag of 1.

    Parameters
    ----------
    nx : int
        Number of pixels in the x-dimension (width)
    ny : int
        Number of pixels in the y-dimension (height)
    znear : float
        Near clipping plane distance (from eye/camera)
    zfar : float
        Far clipping plane distance (from eye/camera)

    Returns
    -------
    mat : np.ndarray
        A 4x4 affine matrix
    """
    n = znear
    f = zfar

    mat = np.array(
        [
            [1 / (nx / ny), 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2.0 / (n - f), (f + n) / (n - f)],
            [0, 0, 0, 1],
        ]
    )
    return mat


def crop_matrix_to_3d(mat_33):
    """Transforms a 3x3 matrix used for cropping (on 2D coordinates)
    into a 4x4 matrix that can be used to transform 3D vertices.
    It assumes that there is no rotation element.

    Parameters
    ----------
    mat_33 : np.ndarray
        A 3x3 affine matrix

    Returns
    -------
    mat_44 : np.ndarray
        A 4x4 affine matrix
    """
    # Define translation in x, y, & z (z = 0)
    t_xyz = np.r_[mat_33[:2, 2], 0]

    # Add column representing z at the diagonal
    mat_44 = np.c_[mat_33[:, :2], [0, 0, 1]]

    # Add back translation
    mat_44 = np.c_[mat_44, t_xyz]

    # Make it a proper 4x4 matrix
    mat_44 = np.r_[mat_44, [[0, 0, 0, 1]]]

    return mat_44

import torch
import numpy as np
from scipy.spatial import Delaunay
        

def create_viewport_matrix(nx, ny, device='cuda'):
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

    mat = torch.tensor(
        [
            [nx / 2, 0, 0, (nx - 1) / 2],
            [0, -ny / 2, 0, (ny - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device
    )

    return mat


def create_ortho_matrix(nx, ny, znear=0.05, zfar=100.0, device='cuda'):
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

    mat = torch.tensor(
        [
            [1 / (nx / ny), 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2.0 / (n - f), (f + n) / (n - f)],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device
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

    # Infer device from input (should always be the same)
    device = mat_33.device

    # Define translation in x, y, & z (z = 0)
    b = mat_33.shape[0]
    t_xyz = torch.cat([mat_33[:, :2, 2], torch.zeros((b, 1), device=device)], dim=1)
    
    # Add column representing z at the diagonal
    mat_33 = torch.cat([mat_33[:, :, :2],
                        torch.tensor([0, 0, 1], device=device).repeat(b, 1)[:, :, None]], dim=2)

    # Add back translation
    mat_34 = torch.cat([mat_33, t_xyz.unsqueeze(2)], dim=2)

    # Make it a proper 4x4 matrix
    mat_44 = torch.cat([mat_34, torch.tensor([0, 0, 0, 1], device=device).repeat(b, 1, 1)], dim=1)
    return mat_44



def apply_perspective_projection(v, mat):
    """" Applies a perspective projection of ``v`` into NDC space. 
    
    Parameters
    ----------
    v : np.ndarray
        A 2D (vertices x XYZ) array with vertex data
    mat : np.ndarray
        A 4x4 perspective projection matrix
    """
    
    v_proj = np.c_[v, np.ones(v.shape[0])] @ mat.T
    v_proj /= v_proj[:, 3, None]
    
    return v_proj


def embed_points_in_mesh(v, f, p, ):
    """ Embed points in an existing mesh by finding the face it is contained in and
    computing its barycentric coordinates. Works with either 2D or 3D data.
    
    Parameters
    ----------
    v : np.ndarray
        Vertices of the existing mesh (a 2D vertices x [2 or 3] array)
    f : np.ndarray
        Faces (polygons) of the existing mesh (a 2D faces x [2 or 3] array)
    p : np.ndarray
        Points (vertices) to embed (a 2D vertices x [2 or 3] array)

    Returns
    -------
    triangles : np.ndarray
        A 1D array with faces corresponding to the vertices of ``p``
    bcoords : np.ndarray
        A 2D array (vertices x 3) array with barycentric coordinates
    """

    # Replace existing Delaunay triangulation with faces (`f`) 
    dim = p.shape[1]
    tri = Delaunay(v[:, :dim])
    tri.simplices = f.astype(np.int32)
    
    # `find_simplex` returns, for each vertex in `p`, the triangle it is contained in
    triangles = tri.find_simplex(p)

    # For some reason, sometimes it returns an index larger than the number of faces
    triangles[triangles >= f.shape[0]] = -1

    # Compute barycentric coordinates; adapted from:
    # https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
    X = tri.transform[triangles, :2]
    Y = p - tri.transform[triangles, 2]
    b = np.einsum('ijk,ik->ij', X, Y)
    bcoords = np.c_[b, 1 - b.sum(axis=1)]

    # Some vertices in `p` fall outside the mesh (i.e., not contained in a triangle),
    # and get the index `-1`
    outside = np.where(triangles == -1)[0]
    for i in outside:
        # Find the closest vertex (euclidean distance)
        v_closest = np.argmin(((p[i, :] - v) ** 2).sum(axis=1))
        
        # alternative: nearest neighbors
        # (1 / dist) / (1 / dist).sum()
        
        # Find the face(s) in which this vertex is contained
        f_idx, v_idx = np.where(f == v_closest)

        # Pick (for no specific reason) the first triangle to be one its contained in
        triangles[i] = f_idx[0]
        
        # Set the barycentric coordinates such that it is 1 for the closest vertex and
        # zero elsewhere
        bcoords[i, :] = np.zeros(3)
        bcoords[i, v_idx[0]] = 1
    
    return triangles, bcoords


def project_points_from_embedding(v, f, triangles, bcoords):
    """ Project points (vertices) from an existing embedding into a different space.
    
    Parameters
    ----------
    v : np.ndarray
        Points (vertices) to project (:math:`N \times 3`)
    f : np.ndarray
        Faces of original mesh
    triangles : np.ndarray

    """
    vf = v[f[triangles]]  # V x 3 x [2 or 3]
    proj = (vf * bcoords[:, :, None]).sum(axis=1)  # V x [2 or 3]
    return proj
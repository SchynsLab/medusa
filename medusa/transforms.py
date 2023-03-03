"""Module with functions to implement some often used transformations in
Medusa."""

import math
import numpy as np
import torch
from kornia.geometry.transform import resize
from scipy.spatial import Delaunay

from .defaults import DEVICE


def create_viewport_matrix(nx, ny, device=DEVICE):
    """Creates a viewport matrix that transforms vertices in NDC [-1, 1] space
    to viewport (screen) space. Based on a blogpost by Mauricio Poppe:
    https://www.mauriciopoppe.com/notes/computer-graphics/viewing/viewport-
    transform/ except that I added the minus sign at [1, 1], which makes sure
    that the viewport (screen) space origin is in the top left.

    Parameters
    ----------
    nx : int
        Number of pixels in the x dimension (width)
    ny : int
        Number of pixels in the y dimension (height)

    Returns
    -------
    mat : torch.tensor
        A 4x4 tensor representing the viewport transform
    """

    mat = torch.tensor(
        [
            [nx / 2, 0, 0, (nx - 1) / 2],
            [0, -ny / 2, 0, (ny - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )

    return mat


def create_ortho_matrix(nx, ny, znear=0.05, zfar=100.0, device=DEVICE):
    """Creates an orthographic projection matrix, as used by EMOCA/DECA. Based
    on the pyrender implementation. Assumes an xmag and ymag of 1.

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
    mat : torch.tensor
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
        device=device,
    )
    return mat


def crop_matrix_to_3d(mat_33):
    """Transforms a 3x3 matrix used for cropping (on 2D coordinates) into a 4x4
    matrix that can be used to transform 3D vertices. It assumes that there is
    no rotation element.

    Parameters
    ----------
    mat_33 : torch.tensor
        A 3x3 affine matrix

    Returns
    -------
    mat_44 : torch.tensor
        A 4x4 affine matrix
    """

    # Infer device from input (should always be the same)
    device = mat_33.device

    # Define translation in x, y, & z (z = 0)
    b = mat_33.shape[0]
    t_xyz = torch.cat([mat_33[:, :2, 2], torch.zeros((b, 1), device=device)], dim=1)

    # Add column representing z at the diagonal
    mat_33 = torch.cat(
        [
            mat_33[:, :, :2],
            torch.tensor([0, 0, 1], device=device).repeat(b, 1)[:, :, None],
        ],
        dim=2,
    )

    # Add back translation
    mat_34 = torch.cat([mat_33, t_xyz.unsqueeze(2)], dim=2)

    # Make it a proper 4x4 matrix
    mat_44 = torch.cat(
        [mat_34, torch.tensor([0, 0, 0, 1], device=device).repeat(b, 1, 1)], dim=1
    )
    return mat_44


def apply_perspective_projection(v, mat):
    """" Applies a perspective projection of ``v`` into NDC space.

    Parameters
    ----------
    v : torch.tensor
        A 2D (vertices x XYZ) tensor with vertex data
    mat : torch.tensor
        A 4x4 perspective projection matrix
    """

    v_proj = np.c_[v, np.ones(v.shape[0])] @ mat.T
    v_proj /= v_proj[:, 3, None]

    return v_proj


def embed_points_in_mesh(v, f, p):
    """Embed points in an existing mesh by finding the face it is contained in
    and computing its barycentric coordinates. Works with either 2D or 3D data.

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
    b = np.einsum("ijk,ik->ij", X, Y)
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
    """Project points (vertices) from an existing embedding into a different
    space.

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


def estimate_similarity_transform(src, dst, estimate_scale=True):
    """Estimate a similarity transformation matrix for two batches of points
    with N observations and M dimensions; reimplementation of the ``_umeyama``
    function of the scikit-image package.

    Parameters
    ----------
    src : torch.Tensor
        A tensor with shape batch_size x N x M
    dst : torch.Tensor
        A tensor with shape batch_size x N x M
    estimate_scale : bool
        Whether to also estimate a scale parameter

    Raises
    ------
    ValueError
        When N (number of points) < M (number of dimensions)
    """
    batch, num, dim = src.shape[:3]

    if num < dim:
        raise ValueError(
            "Cannot compute (unique) transform with fewer "
            f"points ({num}) than dimensions ({dim})!"
        )

    device = src.device
    if dst.device != device:
        dst = dst.to(device)

    if dst.ndim == 2:
        dst = dst.repeat(batch, 1, 1)

    # Compute mean of src and dst.
    src_mean = src.mean(axis=1).unsqueeze(dim=1)
    dst_mean = dst.mean(axis=1).unsqueeze(dim=1)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.mT @ src_demean / num

    # # Eq. (39).
    d = torch.ones((batch, dim), dtype=torch.float32, device=device)
    d[torch.linalg.det(A) < 0, dim - 1] = -1

    T = torch.eye(dim + 1, dtype=torch.float32, device=device).repeat(batch, 1, 1)
    U, S, V = torch.linalg.svd(A)

    # # Eq. (40) and (43).
    rank = torch.linalg.matrix_rank(A)
    T[rank == 0, ...] *= torch.nan

    det_U = torch.linalg.det(U)
    det_V = torch.linalg.det(V)

    idx = torch.logical_and(rank == dim - 1, det_U * det_V > 0)
    if idx.sum() > 0:
        T[idx, :dim, :dim] = U[idx, ...] @ V[idx, ...]

    idx = torch.logical_and(rank == dim - 1, det_U * det_V < 0)
    if idx.sum() > 0:
        s = d[idx, dim - 1]
        d[idx, dim - 1] = -1
        T[idx, :dim, :dim] = U[idx, ...] @ torch.diag_embed(d[idx, :]) @ V[idx, ...]
        d[idx, dim - 1] = s

    idx = torch.logical_and(rank != 0, rank != (dim - 1))
    if idx.sum() > 0:
        T[idx, :dim, :dim] = U[idx, ...] @ torch.diag_embed(d[idx, ...]) @ V[idx, ...]

    if estimate_scale:
        # Eq. (41) and (42).
        src_var = src_demean.var(axis=1, unbiased=False)
        scale = 1.0 / src_var.sum(axis=1) * (S * d).sum(axis=1)
    else:
        scale = torch.ones(batch, device=device)

    dst_mean = dst_mean.squeeze(dim=1)
    src_mean = src_mean.squeeze(dim=1)
    T[:, :dim, dim] = dst_mean - scale.unsqueeze(1) * (
        T[:, :dim, :dim] @ src_mean.unsqueeze(2)
    ).squeeze(dim=2)
    T[:, :dim, :dim] *= scale[:, None, None]

    return T


def resize_with_pad(imgs, output_size=(224, 224), out_dtype=torch.uint8, **kwargs):
    """Resizes image with right-bottom padding (with zeros), as used in
    insightface's ``SCRFD`` detector.

    Parameters
    ----------
    imgs : torch.tensor
        Tensor of shape b x 3 x h x w (batch dimension is optional)
    output_size : tuple
        Desired output shape (width, heigth)
    out_dtype : torch dtype
        Output datatype
    kwargs : dict
        Keyword arguments passed to ``kornia.geometry.transform.resize``

    Returns
    -------
    imgs : torch.tensor
        Resized images of shape b x 3 x new_h x new_w
    """
    ih, iw = imgs.shape[-2:]
    im_ratio = ih / iw

    mw, mh = output_size
    model_ratio = mh / mw

    if im_ratio > model_ratio:
        h = mh
        w = h / im_ratio
    else:
        w = mw
        h = w * im_ratio

    out_shape = list(imgs.shape)
    out_shape[-2:] = [mh, mw]

    h, w = int(h), int(w)
    imgs_resized = torch.zeros(out_shape, device=imgs.device, dtype=out_dtype)
    imgs_resized[..., :h, :w] = resize(imgs, (h, w), **kwargs).to(out_dtype)
    del imgs  # free up space

    scale = h / ih
    return imgs_resized, scale


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis
    sequence."""

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def compose_matrix(scale=None, shear=None, angles=None, translate=None,
                   perspective=None):
    """Return transformation matrix from sequence of transformations."""
    M = np.identity(4)
    if perspective is not None:
        P = np.identity(4)
        P[3, :] = perspective[:4]
        M = np.dot(M, P)
    if translate is not None:
        T = np.identity(4)
        T[:3, 3] = translate[:3]
        M = np.dot(M, T)

    if angles is not None:
        R = euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
        M = np.dot(M, R)

    if shear is not None:
        Z = np.identity(4)
        Z[1, 2] = shear[2]
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = np.dot(M, Z)
    if scale is not None:
        S = np.identity(4)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = np.dot(M, S)
    M /= M[3, 3]

    return M


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis."""

    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def decompose_matrix(matrix):
    """Return sequence of transformations from transformation matrix."""

    M = np.array(matrix, dtype=np.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0
    if not np.linalg.det(P):
        raise ValueError("matrix is singular")

    scale = np.zeros((3, ))
    shear = [0.0, 0.0, 0.0]
    angles = [0.0, 0.0, 0.0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0.0, 0.0, 0.0, 1.0
    else:
        perspective = np.array([0.0, 0.0, 0.0, 1.0])

    translate = M[3, :3].copy()
    M[3, :3] = 0.0

    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        np.negative(scale, scale)
        np.negative(row, row)

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective

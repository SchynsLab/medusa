import torch
from pathlib import Path

import numpy as np
from skimage.transform._geometric import _umeyama
from tqdm import tqdm
from trimesh.registration import icp, procrustes
from trimesh.transformations import compose_matrix, decompose_matrix, transform_points

from ..containers import Data4D


def align(
    data,
    algorithm="icp",
    additive_alignment=False,
    ignore_existing=False,
    reference_index=0,
):
    """Aligment of 3D meshes over time.

    Parameters
    ----------
    data : str, Data
        Either a path (``str`` or ``pathlib.Path``) to a ``medusa`` hdf5
        data file or a ``Data`` object (like ``FlameData`` or ``MediapipeData``)
    algorithm : str
        Either 'icp' or 'umeyama'; ignored for Mediapipe or EMOCA reconstructions
        (except if ``additive_alignment`` or ``ignore_existing`` is set to ``True``)
    additive_alignment : bool
        Whether to estimate an additional set of alignment parameters on
        top of the existing ones (if present; ignored otherwise)
    ignore_existing : bool
        Whether to ignore the existing alignment parameters
    reference_index : int
        Index of the mesh used as the reference mesh; for reconstructions that already
        include the local-to-world matrix, the reference mesh is only used to fix the
        camera to; for other reconstructions, the reference mesh is used as the target
        to align all other meshes to

    Returns
    -------
    data : medusa.core.*Data
        An object with a class inherited from ``medusa.core.BaseData``

    Examples
    --------
    Align sequence of 3D Mediapipe meshes using its previously estimated local-to-world
    matrices (the default alignment option):

    >>> from medusa.data import get_example_h5
    >>> data = get_example_h5(load=True, model='mediapipe')
    >>> data.space  # before alignment, data is is 'world' space
    'world'
    >>> data = align(data)
    >>> data.space  # after alignment, data is in 'local' space
    'local'

    Do an initial alignment of EMOCA meshes using the existing transform, but also
    do additional alignment (probably not a good idea):

    >>> data = get_example_h5(load=True, model='emoca-coarse')
    >>> data = align(data, algorithm='icp', additive_alignment=True)
    """

    if isinstance(data, (str, Path)):
        data = Data4D.load(data)

    if additive_alignment and ignore_existing:
        raise ValueError("Cannot do additive alignment and ignore existing!")

    if data.space == "local" and not additive_alignment and not ignore_existing:
        raise ValueError(
            "Data is already in local space! No need to align; "
            "If you want to perform alignment on top of the existing "
            "local-to-world matrix, set `additive_alignment` to True; "
            "If you want to ignore the existing matrix, set "
            "`ignore_existing` to True!"
        )

    # vidx represents the index of the vertices that we'll use for
    # alignment (using *all* vertices is not a good idea, as it may
    # inadvertently try to align non-rigid motion)

    topo = data._infer_topo()
    if topo == "mediapipe":
        # Technically not necessary anymore, because we have the model parameters
        # fmt: off
        v_idx = [389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,  # contour
                 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162,  # contour
                 94, 19, 1, 4, 5, 195, 197, 6]  # nose ridge
        # fmt: on
    elif topo == "flame":
        v_idx = np.load(Path(__file__).parents[1] / "data/flame/scalp_flame.npy")
    else:
        raise ValueError("Unknown reconstruction model!")

    # Are we going to use the existing alignment parameters (local-to-world)?
    if data.mat is not None and not ignore_existing:

        for i in range(data.v.shape[0]):
            # Use inverse of local-to-world matrix (i.e., world-to-local)
            # to 'remove' global motion
            this_mat = np.linalg.inv(data.mat[i, ...].cpu().numpy())
            data.v[i, ...] = torch.as_tensor(
                transform_points(data.v[i, ...].cpu().numpy(), this_mat),
                device=data.device,
            )

    # Because the object space changed (world --> local) and we assume
    # the object will be visualized on top of the video frames, we need
    # to update the camera matrix with the world-to-local matrix; this ensure
    # the rendering will be in the "right" pixel space; note that we only do
    # this for the first frame (otherwise we 're-introduce' the global motion,
    # but this time in the camera, and we assume a fixed camera of course)
    data.cam_mat = np.linalg.inv(data.mat[reference_index, ...]) @ data.cam_mat
    data.space = "local"

    return data


def _icp(source, target, scale=True, ignore_shear=True):
    """ICP implementation from trimesh."""
    # First, do a rough procrustus reg, followed by icp (recommended by trimesh)
    regmat, _, _ = procrustes(source, target, reflection=False, scale=scale)

    # Stupic hack: set shear to None and recompose matrix
    if ignore_shear:
        scale_, _, angles, translate, _ = decompose_matrix(regmat)
        regmat = compose_matrix(scale_, None, angles, translate)

    # Run ICP with `regmat` as initial matrix
    regmat, _, _ = icp(source, target, initial=regmat, reflection=False, scale=scale)

    # Remove shear again
    if ignore_shear:
        scale_, _, angles, translate, _ = decompose_matrix(regmat)
        regmat = compose_matrix(scale_, None, angles, translate)

    return regmat


import numpy as np


def rigid_transform_3D(A, B):
    # From https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

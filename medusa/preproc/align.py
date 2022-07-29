import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from trimesh.registration import icp, procrustes
from skimage.transform._geometric import _umeyama
from trimesh.transformations import decompose_matrix, transform_points, compose_matrix

from ..io import load_h5


def align(data, algorithm='icp', additive_alignment=False, ignore_existing=False,
          reference_index=0):
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
    
    Align sequence of 3D FAN meshes using ICP:
    
    >>> data = get_example_h5(load=True, model='fan')
    >>> data.mat is None  # no affine matrices yet
    True
    >>> data = align(data, algorithm='icp')
    >>> data.mat.shape  # an affine matrix for each time point!
    (232, 4, 4)
    
    Do an initial alignment of EMOCA meshes using the existing transform, but also
    do additional alignment (probably not a good idea):
    
    >>> data = get_example_h5(load=True, model='emoca-coarse')
    >>> data = align(data, algorithm='icp', additive_alignment=True)
    """

    if isinstance(data, (str, Path)):
        data = load_h5(data)

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
    if data.recon_model_name == "fan":
        v_idx = range(17)  # contour only
    elif data.recon_model_name == "mediapipe":
        # Technically not necessary anymore, because we have the model parameters
        v_idx = [389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,  # contour
                 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162,  # contour
                 94, 19, 1, 4, 5, 195, 197, 6]  # nose ridge
    elif data.recon_model_name in ["emoca-coarse", "deca-dense"]:
        v_idx = np.load(Path(__file__).parents[1] / 'data/scalp_flame.npy') 
    #else:
    #    raise ValueError("Unknown reconstruction model!")

    if data.logger.level <= logging.INFO:
        desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ]  Align frames")
        iter_ = tqdm(range(data.v.shape[0]), desc=desc)
    else:
        iter_ = range(data.v.shape[0])

    # Are we going to use the existing alignment parameters (local-to-world)?
    if data.mat is not None and not ignore_existing:

        for i in iter_:
            # Use inverse of local-to-world matrix (i.e., world-to-local)
            # to 'remove' global motion
            this_mat = np.linalg.inv(data.mat[i, ...])
            data.v[i, ...] = transform_points(data.v[i, ...], this_mat)

    # Are we going to do (additional) data-driven alignment?
    if data.mat is None or ignore_existing or additive_alignment:

        if data.mat is None:
            # Initialize with identity matrices
            T = data.v.shape[0]
            data.mat = np.repeat(np.eye(4)[None, :, :], T, axis=0)

        # Set target to be the reference index; default is 0, this is an arbitrary choice,
        # but common in e.g. fMRI motion correction
        target = data.v[reference_index, v_idx, :]

        for i in iter_:

            if additive_alignment:
                orig_mat = data.mat[i, ...]
            else:
                orig_mat = np.eye(4)

            # Code below does some funky iterative alignment, doesn't work as
            # well as I hoped

            # regmat = np.eye(4)
            # source = data.v[i, :, :].copy()
            # for ii in reversed(range(0, i)):
            #     target = data.v[ii, :, :]
            #     r = _icp(source, target)
            #     #source = transform_points(source, r)
            #     regmat = r @ regmat

            # v[i, :, :] = transform_points(data.v[i, :, :], regmat)#source

            source = data.v[i, v_idx, :]  # to be aligned
            if i == 0:
                # First mesh does not have to be aligned
                mat_ = np.eye(4)
            else:
                if algorithm == "icp":  # trimesh ICP implementation
                    mat_ = _icp(source, target, scale=True, ignore_shear=False)
                else:  # scikit-image 3D umeyama similarity transform
                    mat_ = _umeyama(source, target, estimate_scale=True)

            # Apply estimated world-to-local matrix and update existing
            # local-to-world matrix (data.mat) if it exists
            data.v[i, ...] = transform_points(data.v[i, ...], mat_)
            data.mat[i, ...] = np.linalg.inv(mat_) @ orig_mat

    data.v = data.v.astype(np.float32)

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

    
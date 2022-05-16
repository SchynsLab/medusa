import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from trimesh.registration import icp, procrustes
from skimage.transform._geometric import _umeyama
from trimesh.transformations import decompose_matrix, transform_points, compose_matrix

from ..data import load_h5
from ..utils import get_logger

logger = get_logger()


def align(data, algorithm, qc=False, additive_alignment=False, ignore_existing=False):
    """ Aligment of 3D meshes over time. 
    
    Parameters
    ----------
    data : str, Data
        Either a path (str, pathlib.Path) to a `gmfx` hdf5 data file
        or a gmfx.io.Data object (i.e., data loaded from the hdf5 file)
    algorithm : str
        Either 'icp' or 'umeyama'
    qc : bool
        Whether to visualize a quality control plot
    additive_alignment : bool
        Whether to estimate an additional set of alignment parameters on
        top of the existing ones (if present; ignored otherwise)
    ignore_existing : bool
        Whether to ignore the existing alignment parameters
    """

    if isinstance(data, (str, Path)):
        logger.info(f"Loading data from {data} ...")
        data = load_h5(data)

    if data.space == 'local' and not additive_alignment and not ignore_existing:
        raise ValueError("Data is already in local space! No need to align; "
                         "If you want to perform alignment on top of the existing "
                         "local-to-world matrix, set `additive_alignment` to True; "
                         "If you want to ignore the existing matrix, set "
                         "`ignore_existing` to True!")

    # vidx represents the index of the vertices that we'll use for
    # alignment (using *all* vertices is not a good idea, as it may
    # inadvertently try to align non-rigid motion)    
    if data.recon_model_name == 'FAN-3D':
        v_idx = range(17)  # contour only
    elif data.recon_model_name == 'mediapipe':
        # Technically not necessary anymore, because we have the model parameters
        v_idx = [389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,    # contour
                 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, # contour
                 94, 19, 1, 4, 5, 195, 197, 6]  # nose ridge                
    elif data.recon_model_name == 'emoca':
        import pickle
        with open('FLAME_masks.pkl', 'rb') as f_in:
            vidx = pickle.load(f_in, encoding='latin1')['scalp']
    else:
        raise ValueError("Unknown reconstruction model!")

    # Are we going to use the existing alignment parameters (local-to-world)?
    if data.mat is not None and not ignore_existing:
        
        desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ]  Align frames')        
        for i in tqdm(range(data.v.shape[0]), desc=desc):
            # Use inverse of local-to-world matrix (i.e., world-to-local)
            # to 'remove' global motion
            this_mat = np.linalg.inv(data.mat[i, ...])
            data.v[i, ...] = transform_points(data.v[i, ...], this_mat)

    # Are we going to do (additional) data-driven alignment?
    if data.mat is None or ignore_existing or additive_alignment:

        # Set target to be the first timepoint; this is an arbitrary choice,
        # but common in e.g. fMRI motion correction
        target = data.v[0, vidx, :]
    
        desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ]  Align frames')
        for i in tqdm(range(data.v.shape[0]), desc=desc):

            if data.mat is not None:
                # What do we use as the initial/original matrix?
                if ignore_existing:
                    orig_mat = np.eye(4)
                elif additive_alignment:
                    # We estimate a new matrix on top of existing one!
                    orig_mat = data.mat[i, ...]
                else:
                    orig_mat = np.eye(4)
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
                if algorithm == 'icp':  # trimesh ICP implementation
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
    data.cam_mat = np.linalg.inv(data.mat[0, ...]) @ data.cam_mat

    # Save!
    pth = data.path
    desc = 'desc-' + pth.split('desc-')[1].split('_')[0] + '+align'
    f_out = pth.split('desc-')[0] + desc
    
    if qc:
        data.plot_data(f_out + '_qc.png', plot_motion=True, plot_pca=True, n_pca=3)
    
    data.space = 'local'
    data.save(f_out + '_shape.h5')


def _icp(source, target, scale=True, ignore_shear=True):
    """ ICP implementation from trimesh. """    
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

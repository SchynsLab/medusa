import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from trimesh.registration import icp, procrustes
from trimesh.transformations import decompose_matrix, transform_points, compose_matrix
from skimage.transform._geometric import _umeyama

from ..io import load_h5
from ..utils import get_logger

logger = get_logger()


def align(data, algorithm, video):
    """ Aligment of 3D meshes over time. 
    
    Parameters
    ----------
    data : str, Data
        Either a path (str, pathlib.Path) to a `gmfx` hdf5 data file
        or a gmfx.io.Data object (i.e., data loaded from the hdf5 file)
    algorithm : str
        Either 'icp' or 'umeyama'
    """

    if isinstance(data, (str, Path)):
        # if data is a path to a hdf5 file, load it
        # (used by CLI)
        logger.info(f"Loading data from {data} ...")
        data = load_h5(data)
    
    if data.recon_model_name == 'FAN-3D':
        vidx = range(17)  # contour only
    elif data.recon_model_name == 'mediapipe':
        vidx = [389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,    # contour
                152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, # contour
                94, 19, 1, 4, 5, 195, 197, 6]  # nose ridge                
    else:
        vidx = range(data.v.shape[1])
    
    # Set target to be the first timepoint
    target = data.v[0, vidx, :]
    T = data.v.shape[0]

    # Pre-allocate registration parameters (save for 
    # nuisance parameters in model fitting)
    reg_params = np.zeros((T, 12))

    # Loop over meshes
    desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ] ')
    v = np.zeros_like(data.v)
    
    mat = np.zeros((data.v.shape[0], 4, 4))
    for i in tqdm(range(T), desc=f'{desc} Align frames'):
        
        if data.mat is not None:
            # If there is already a world-to-local matrix,
            # use its inverse to remove any global motion
            vh = np.c_[data.v[i, ...], np.ones(data.v.shape[1])]
            v[i, ...] = (vh @ np.linalg.inv(data.mat[i, :, :]).T)[:, :3]

            # We want to keep the original z-coordinate from the first frame,
            # otherwise we need to move the camara
            v[i, :, 2] = data.v[0, :, 2]
            mat[i, ...] = data.mat[i, :, :]
            continue
        
        # regmat = np.eye(4)
        # source = data.v[i, :, :].copy()
        # for ii in reversed(range(0, i)):
        #     target = data.v[ii, :, :]
        #     r = _icp(source, target)
        #     #source = transform_points(source, r)
        #     regmat = r @ regmat

        # v[i, :, :] = transform_points(data.v[i, :, :], regmat)#source

        source = data.v[i, vidx, :]  # to be aligned
        if i == 0:
            # First mesh does not have to be aligned
            mat_ = np.eye(4)
        else:
            if algorithm == 'icp':  # trimesh ICP implementation
                mat_ = _icp(source, target, scale=True, ignore_shear=False)
            else:  # scikit-image 3D umeyama similarity transform
                mat_ = _umeyama(source, target, estimate_scale=True)

        # Apply estimated transformation            
        v[i, ...] = transform_points(data.v[i, ...], mat_)
        mat[i, ...] = np.linalg.inv(mat_)
        
    data.v = v.astype(np.float32)
    data.mat = mat

    # Save!
    pth = data.path
    desc = 'desc-' + pth.split('desc-')[1].split('_')[0] + '+align'
    f_out = pth.split('desc-')[0] + desc
    data.plot_data(f_out + '_qc.png', plot_motion=True, plot_pca=True, n_pca=3)
    
    data.render_video(f_out + '_shape.gif', video=video)
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
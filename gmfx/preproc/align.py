import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from trimesh.registration import icp, procrustes
from trimesh.transformations import decompose_matrix, transform_points, compose_matrix
from skimage.transform._geometric import _umeyama as _umeyama_skimage

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
        vidx = range(17)
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
    for i in tqdm(range(T), desc=f'{desc} Align frames'):
 
        source = data.v[i, vidx, :]  # to be aligned
        if i == 0:
            # First mesh does not have to be aligned
            regmat = np.eye(4)
        else:
            if algorithm == 'icp':  # trimesh ICP implementation
                regmat = _icp(source, target, scale=True, ignore_shear=False)
            else:  # scikit-image 3D umeyama similarity transform
                regmat = _umeyama_skimage(source, target, estimate_scale=True)

            # Apply estimated transformation            
            data.v[i, ...] = transform_points(data.v[i, ...], regmat)

        # Decompose matrix into reg parameters and store
        scale, shear, angles, translate, _ = decompose_matrix(regmat)
        reg_params[i, :] = np.r_[angles, translate, scale, shear]

    data.v = data.v.astype(np.float32)
    data.motion = reg_params
    #data.motion_cols = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z',
    #                    'scale_x', 'scale_y', 'scale_z', 'shear_x', 'shear_y', 'shear_z']

    # Save!
    pth = data.path
    desc = 'desc-' + pth.split('desc-')[1].split('_')[0] + '+align'
    f_out = pth.split('desc-')[0] + desc
    data.render_video(f_out + '_shape.gif', video=video)
    data.plot_data(f_out + '_qc.png', plot_motion=True, plot_pca=True, n_pca=3)
    data.save(f_out + '_shape.h5')


def _icp(source, target, scale=True, ignore_shear=True):
    """ ICP implementation from trimesh. """    
    # First, do a rough procrustus reg, followed by icp (recommended by trimesh)
    regmat, _, _ = procrustes(source, target, reflection=False, scale=scale)

    # Stupic hack: set shear to None and recompose matrix
    if ignore_shear:
        scale, _, angles, translate, _ = decompose_matrix(regmat)
        regmat = compose_matrix(scale, None, angles, translate)

    # Run ICP with `regmat` as initial matrix
    regmat, _, _ = icp(source, target, initial=regmat, reflection=False, scale=scale)

    # Remove shear again
    if ignore_shear:
        scale, _, angles, translate, _ = decompose_matrix(regmat)
        regmat = compose_matrix(scale, None, angles, translate)

    return regmat
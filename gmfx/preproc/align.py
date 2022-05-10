import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pyrender import OrthographicCamera
from trimesh.registration import icp, procrustes
from skimage.transform._geometric import _umeyama
from trimesh.transformations import decompose_matrix, transform_points, compose_matrix

from ..data import load_h5
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
    video : str
        Path to video to render reconstruction on top of
        (optional)
    """

    if isinstance(data, (str, Path)):
        # if data is a path to a hdf5 file, load it
        # (used by CLI)
        logger.info(f"Loading data from {data} ...")
        data = load_h5(data)

    # vidx represents the index of the vertices that we'll use for
    # alignment (using *all* vertices is not a good idea)    
    if data.recon_model_name == 'FAN-3D':
        vidx = range(17)  # contour only
    elif data.recon_model_name == 'mediapipe':
        # Technically not necessary anymore, because we have the model parameters
        vidx = [389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,    # contour
                152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, # contour
                94, 19, 1, 4, 5, 195, 197, 6]  # nose ridge                
    else:
        # Just use everyt
        vidx = range(data.v.shape[1])
    
    # Set target to be the first timepoint
    target = data.v[0, vidx, :]
    T = data.v.shape[0]

    # Loop over meshes
    desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ] ')    
    reference_z = data.v[0, :, 2].copy()  # used later
    for i in tqdm(range(T), desc=f'{desc} Align frames'):
        
        # Is there a cropping matrix (like for EMOCA)?
        if getattr(data, 'cropmat', None) is not None:
            # Undo the cropping transformation by mapping the vertices to
            # raster space, applying the transform, and mapping it back to world space
            # (Also, a new image size can be give; here: (256, 256))
            data.v[i, ...] = _undo_cropmat(
                data.v[i, ...], data.cropmat[i, ...],
                data.img_size, (256, 256)
            )

        if data.mat is not None:
            # If there is already a world-to-local matrix (like EMOCA and Mediapipe)
            # use its inverse to remove any global motion
            this_mat = np.linalg.inv(data.mat[i, ...])
            data.v[i, ...] = transform_points(data.v[i, ...], this_mat)
            
            # We want to keep the original z-coordinate from the first frame
            data.v[i, :, 2] = reference_z

            if data.recon_model_name == 'emoca':
                # Add back scale (7 is a somewhat arbitrary choice), because
                # otherwise we have to mess with the camera
                #scale = decompose_matrix(data.mat[0, ...])[0]
                #data.v[i, :, :2] *= np.mean(scale)
                data.v[i, :, :2] *= 7

            continue
        
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
        data.v[i, ...] = transform_points(data.v[i, ...], mat_)
        data.mat[i, ...] = np.linalg.inv(mat_)
        
    data.v = data.v.astype(np.float32)

    if getattr(data, 'cropmat', None) is not None:
        data.img_size = (256, 256)

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


def _undo_cropmat(v, mat, img_size, new_img_size):
    """ 'Undoes' the cropping transformation by mapping
    the vertices to raster space, applying the transformation (`mat`,
    the one estimated by FaceAlignment), and reprojecting the vertices
    to world space. Note that we can also choose a new image size
    to render our motion-correction face in.
    
    Parameters
    ----------
    v : np.ndarray
        A 2D array of shape nV (number of verts) x 3 (XYZ)
    mat : np.ndarray
        A 3x3 affine matrix representing the cropping operation
        (from the similarity transform estimated in FAN)
    img_size : tuple[int]
        Original image size (width, height)
    new_img_size : tuple[int]
        New image size (width, height)

    Returns
    -------
    v : np.ndarray
        A 2D array of shape nV (number of verts) x 3 (XYZ),
        but with the cropping operation 'projected out'
    """
    nV = v.shape[0]
    v = v.copy()
    P = OrthographicCamera(1, 1).get_projection_matrix(*img_size)
    v = np.c_[v, np.ones(nV)] @ P.T
    z = v[:, 2].copy()  # save for later
    v = v[:, :2]
    
    # Invert y-axis (because image space)
    v[:, 1] = -v[:, 1]
    
    # Normalize from [-1 , 1] to [0, 1]
    v = (v + 1) / 2
    
    # NDC to raster
    v[:, 0] *= img_size[0]
    v[:, 1] *= img_size[1]

    # Undo cropping transform 
    v = np.c_[v, np.ones(nV)]
    v = (v @ mat.T)[:, :2]

    # Now map back to [0, 1] NDC space
    v[:, 0] = v[:, 0] / 224
    v[:, 1] = v[:, 1] / 224

    # NDC -> image
    v = -(1 - 2 * v)
    v[:, 1] = -v[:, 1]

    # inverted ortho transform (2D -> 3D)
    P = OrthographicCamera(1, 1).get_projection_matrix(new_img_size)
    v = np.c_[v, z, np.ones(nV)] @ np.linalg.inv(P.T)
    v = v[:, :3]
    return v

import os
import numpy as np
import os.path as op
import pandas as pd
from glob import glob
from tqdm import tqdm
from trimesh.registration import icp, procrustes
from trimesh.transformations import decompose_matrix, transform_points, compose_matrix
from skimage.transform._geometric import _umeyama as _umeyama_skimage

from ...viz.viz import images_to_mp4, plot_shape
from ..constants import EYES_NOSE, SCALP


def align(in_dir, participant_label, algorithm, ref_verts, save_all):
    """ Aligment of 3D meshes over time. """
    
    recon_dir = op.join(in_dir, participant_label, 'recon')
    align_dir = op.join(in_dir, participant_label, 'align')
    os.makedirs(align_dir, exist_ok=True)

    # Load (un-aligned) vertices from recon stage
    verts = np.load(op.join(recon_dir, participant_label + '_desc-recon_shape.npy'))

    # Pick reference vertices (to use for aligment)
    if ref_verts == 'all':
        verts4align = np.ones(verts.shape[1], dtype=bool)
    elif ref_verts == 'eyes+nose':
        verts4align = EYES_NOSE
    elif ref_verts == 'scalp':
        verts4align = SCALP
    else:
        pass  # error will be raised earlier
    
    # Set target to be the first timepoint
    # Idea: change to first n aligned timepoints?
    target = verts[0, verts4align, :]

    # Pre-allocate registration parameters (save for 
    # nuisance parameters in model fitting)
    reg_params = np.zeros((verts.shape[0], 12))

    # Loop over meshes
    for i in tqdm(range(verts.shape[0]), desc='Align'):
 
        source = verts[i, verts4align, :]  # to be aligned
        if i == 0:
            # First mesh does not have to be aligned
            regmat = np.eye(4)
        else:
            if algorithm == 'icp':  # trimesh ICP implementation
                _, regmat = _icp(source, target, do_scale=True, ignore_shear=True)
            else:  # scikit-image 3D umeyama similarity transform
                _, regmat = _umeyama(source, target, do_scale=True)

            # Apply estimated transformation            
            verts[i, ...] = transform_points(verts[i, ...], regmat)

        # Decompose matrix into reg parameters and store
        scale, shear, angles, translate, _ = decompose_matrix(regmat)
        reg_params[i, :] = np.r_[scale, shear, angles, translate]

        # Save rendered img to disk
        f_out = op.join(align_dir, participant_label + f'_img-{str(i+1).zfill(5)}_align.png')
        plot_shape(verts[i, ...], f_out=f_out)

    f_out = op.join(align_dir, participant_label + '_desc-align_shape.npy')
    np.save(f_out, verts)

    cols = [f'{p}_{c}' for p in ['scale', 'shear', 'rot', 'trans'] for c in ['x', 'y', 'z']]
    reg_df = pd.DataFrame(reg_params, columns=cols)
    reg_df.to_csv(op.join(align_dir, participant_label + '_desc-align_parameters.csv'), index=False)

    images = sorted(glob(op.join(align_dir, '*_align.png')))
    f_out = op.join(align_dir, participant_label + '_desc-align_shape.mp4')
    images_to_mp4(images, f_out)
    
    if not save_all:
        _ = [os.remove(f) for f in images]


def _icp(source, target, do_scale=True, ignore_shear=True):
    """ ICP implementation from trimesh. """    
    # First, do a rough procrustus reg, followed by icp (recommended by trimesh)
    regmat, _, _ = procrustes(source, target, reflection=False, scale=do_scale)

    # Stupic hack: set shear to None and recompose matrix
    if ignore_shear:
        scale, _, angles, translate, _ = decompose_matrix(regmat)
        regmat = compose_matrix(scale, None, angles, translate)

    # Run ICP with `regmat` as initial matrix
    regmat, _, _ = icp(source, target, initial=regmat, reflection=False, scale=do_scale)

    # Remove shear again
    if ignore_shear:
        scale, _, angles, translate, _ = decompose_matrix(regmat)
        regmat = compose_matrix(scale, None, angles, translate)

    return regmat


def _umeyama(source, target, do_scale=True):
    """ Wrapper around scikit-image umeyama sim transform. """
    regmat = _umeyama_skimage(source, target, estimate_scale=do_scale)
    return regmat
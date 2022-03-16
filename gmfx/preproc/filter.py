import os
import numpy as np
import os.path as op
from glob import glob
from tqdm import tqdm
from nilearn import signal

from ...viz.viz import images_to_mp4, plot_shape


def filter(in_dir, participant_label, low_pass, high_pass, save_all):

    interp_dir = op.join(in_dir, participant_label, 'interpolate')
    filter_dir = op.join(in_dir, participant_label, 'filter')
    os.makedirs(filter_dir, exist_ok=True)

    verts = np.load(op.join(interp_dir, participant_label + '_desc-interp_shape.npy'))
    
    # Flatten vertices to work with signal.clean (which assumes 2D)
    verts = verts.reshape((verts.shape[0], 5023 * 3))

    # Normalize beforehand, but save parameters
    mu, std = verts.mean(axis=0), verts.std(axis=0)
    verts = (verts - mu) / std
    verts = signal.clean(verts, detrend=True, standardize=False, t_r=0.089, high_pass=high_pass, low_pass=low_pass)

    # Undo normalization to get data back on original scale
    verts = (verts * std + mu).reshape((verts.shape[0], 5023, 3))

    f_out = op.join(filter_dir, participant_label + '_desc-filter_shape.npy')
    np.save(f_out, verts)

    for i in tqdm(range(verts.shape[0]), desc='Plot filter'):
        # Save rendered img to disk
        f_out = op.join(filter_dir, participant_label + f'_img-{str(i+1).zfill(5)}_filter.png')
        plot_shape(verts[i, ...], f_out=f_out)
    
    images = sorted(glob(op.join(filter_dir, '*_filter.png')))
    f_out = op.join(filter_dir, participant_label + '_desc-filter_shape.mp4')
    images_to_mp4(images, f_out)
    
    if not save_all:
        _ = [os.remove(f) for f in images]


if __name__ == '__main__':

    main()
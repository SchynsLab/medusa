import os
import numpy as np
import pandas as pd
import os.path as op
from glob import glob
from tqdm import tqdm
from scipy.interpolate import interp1d

from ..render.utils import images_to_mp4, plot_shape


def interpolate(in_dir, participant_label, save_all):

    align_dir = op.join(in_dir, participant_label, 'align')
    interp_dir = op.join(in_dir, participant_label, 'interpolate')
    os.makedirs(interp_dir, exist_ok=True)

    verts = np.load(op.join(align_dir, participant_label + '_desc-align_shape.npy'))
    
    # TO FIX: bit of a hack to find frametimes
    f_ft = glob(op.join(in_dir, participant_label, 'raw', '*frametimes.tsv'))
    df_ft = pd.read_csv(f_ft[0], sep='\t')
    ft = df_ft['t'].to_numpy()
    sampling_rate = np.mean(np.diff(ft)).round(3)
    new_ft = np.linspace(ft[0], ft[0] + sampling_rate * (ft.size - 1),
                         endpoint=True, num=ft.size)
    
    interpolator = interp1d(ft[:verts.shape[0]], verts, axis=0)
    verts = interpolator(new_ft[:verts.shape[0]])

    f_out = op.join(interp_dir, participant_label + '_desc-interp_shape.npy')
    np.save(f_out, verts)

    df_ft['new_frame_times'] = new_ft
    df_ft.to_csv(f_ft[0], index=False, sep='\t')

    for i in tqdm(range(verts.shape[0]), desc='Plot interp'):
        # Save rendered img to disk
        f_out = op.join(interp_dir, participant_label + f'_img-{str(i+1).zfill(5)}_interp.png')
        plot_shape(verts[i, ...], f_out=f_out)
    
    images = sorted(glob(op.join(interp_dir, '*_interp.png')))
    f_out = op.join(interp_dir, participant_label + '_desc-interp_shape.mp4')
    images_to_mp4(images, f_out)
    
    if not save_all:
        _ = [os.remove(f) for f in images]

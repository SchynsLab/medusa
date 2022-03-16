import os
import cv2
import yaml
import warnings
import shutil
import imageio
import numpy as np
import os.path as op
import pandas as pd
from tqdm import tqdm
from glob import glob

from ..models.deca import DECA
from ..models.deca.utils.config import cfg as deca_cfg
from ..models.tddfa.FaceBoxes.FaceBoxes import FaceBoxes
from ..models.tddfa import TDDFA
from ..models.tddfa import configs

from . import recon_deca, recon_tddfa
from ..io import Data
from ..models.deca.detectors import FAN

warnings.filterwarnings(
    action='ignore',
    category=UserWarning
)

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning
)


def recon(in_dir, out_dir, participant_label, model, device, visualize):
    """ Reconstruction of all frames of a video. """

    data_dir = op.join(in_dir, participant_label)  # to get input data from
    out_dir = op.join(out_dir, participant_label, 'recon')

    if op.isdir(out_dir):
        shutil.rmtree(out_dir)
    else:
        os.makedirs(out_dir)

    if model == 'deca':
        # Init here, otherwise it's done every frame -> slow
        deca = DECA(config=deca_cfg, device=device)
        fan = FAN(device=device)
    else:
        backbone = 'resnet_120x120'
        cfg = yaml.load(open(getattr(configs, backbone)), Loader=yaml.SafeLoader)
        cfg['device'] = device
        face_boxes = FaceBoxes(device=device)
        tddfa = TDDFA(**cfg)

    # Find files and loop over them
    videos = glob(op.join(data_dir, '*_video.mp4'))
    for vid in videos:

        base_f = op.basename(vid).split('.mp4')[0]
        reader = imageio.get_reader(vid)
        fps = reader.get_meta_data()['fps']

        # Use cv2 to get n_frames (not possible with imageio ...)
        cap = cv2.VideoCapture(vid)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ft_path = vid.replace('_video.mp4', '_frametimes.tsv')
        if op.isfile(ft_path):
            frame_t = pd.read_csv(ft_path, sep='\t')['t'].to_numpy()
        else:
            print(f"Warning: did not find frametimes file for video {vid}!")
            frame_t = None

        ev_path = vid.replace('_video.mp4', '_events.tsv')
        if op.isfile(ev_path):
            events = pd.read_csv(ev_path, sep='\t')
        else:
            print(f"Warning: did not find events file for video {vid}!")
            events = None

        if model == 'tddfa':
            pre_v = None

        # Loop over frames in video
        v_4D, i_4D = [], []  # cannot know nr of frames in advance
        for i, frame in tqdm(enumerate(reader), desc=f'Recon {base_f}', total=n_frames):
            
            if model == 'deca':
                v, frame = recon_deca(frame, deca, fan, device=device)
            else:
                frame_bgr = frame[..., ::-1]
                v, pre_v = recon_tddfa(frame_bgr, pre_v, tddfa, face_boxes)

            v_4D.append(v)
            i_4D.append(frame)
    
        reader.close()

        # Stack into 4D arr, create Data obj, and save
        v_4D = np.stack(v_4D)
        i_4D = np.stack(i_4D)

        f_type = 'flame' if model == 'deca' else 'bfm'
        data = Data(
            v=v_4D, imgs=i_4D, frame_t=frame_t, events=events, 
            fps=fps, f_type=f_type, dense=False
        )
        f_out = op.join(out_dir, base_f + '_desc-recon_verts.hdf5')
        data.save(f_out)

        if visualize:
            f_out = f_out.replace('.hdf5', '.mp4')
            data.visualize(f_out)
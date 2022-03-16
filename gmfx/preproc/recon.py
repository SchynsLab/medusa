import cv2
import torch
import warnings
import shutil
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path

from ..detect import FAN
from ..recon import DECA
from ..recon.utils import tensor2image
from ..io import Data


warnings.filterwarnings(
    action='ignore',
    category=UserWarning
)

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning
)


def recon(in_dir, out_dir, participant_label, device, visualize):
    """ Reconstruction of all frames of a video. """

    data_dir = Path(in_dir) / participant_label
    
    if out_dir is None:
        out_dir = Path(in_dir) / 'derivatives' / participant_label / 'recon'
    
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(exist_ok=True, parents=True)

    # Init here, otherwise it's done every frame -> slow
    fan = FAN(device=device)
    deca = DECA(device=device)

    # Find files and loop over them
    for vid in data_dir.glob('*_video.mp4'):

        base_f = vid.stem
        reader = imageio.get_reader(vid)
        fps = reader.get_meta_data()['fps']

        # Use cv2 to get n_frames (not possible with imageio ...)
        cap = cv2.VideoCapture(str(vid))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ft_path = Path(str(vid).replace('_video.mp4', '_frametimes.tsv'))
        if ft_path.exists():
            frame_t = pd.read_csv(ft_path, sep='\t')['t'].to_numpy()
        else:
            print(f"Warning: did not find frametimes file for video {vid}!")
            frame_t = None

        ev_path = Path(str(vid).replace('_video.mp4', '_events.tsv'))
        if ev_path.exists():
            events = pd.read_csv(ev_path, sep='\t')
        else:
            print(f"Warning: did not find events file for video {vid}!")
            events = None

        f_out = str(out_dir / base_f) + '_desc-recon'
        v_4D = np.zeros((n_frames, 5023, 3))        
        writer = imageio.get_writer(f_out + '_viz-orig_shape.mp4', mode='I', fps=fps)
        
        for i, frame in tqdm(enumerate(reader), desc=f'Recon {base_f}', total=n_frames):

            # Prepare frame to be used as background during rendering            
            orig_size = frame.shape[:2]
            frame_orig = (frame / 255.).transpose(2, 0, 1)
            frame_orig = torch.tensor(frame_orig).float()[None, ...].to('cuda')
    
            # Crop image and reconstruct
            frame_cropped = fan(frame)
            enc_dict = deca.encode(frame_cropped)
            
            if i == 0:
                # Initialize moving average
                shape_ma = enc_dict['shape'].clone()
            elif i < 10:
                # First 10 frames, the moving average is updated
                shape_ma = (shape_ma * (i + 1) + enc_dict['shape']) / (i + 2)
            else:
                # After 10 frames, we'll use the current shape_ma
                pass            

            # Set shape to moving average
            enc_dict['shape'] = shape_ma.clone()

            # Decode and render
            dec_dict = deca.decode(enc_dict, tform=fan.tform_params, orig_size=orig_size)
            rend_dict = deca.render_dec(enc_dict, dec_dict, render_world=False,
                                        img_orig=frame_orig)

            enc_dict['pose'][0, :3] = 0
            dec_dict = deca.decode(enc_dict, tform=None)
            v_4D[i, ...] = dec_dict['V'][0].cpu().detach().numpy()
            
            img = tensor2image(rend_dict['shape_detail'][0], bgr=False)
            writer.append_data(img)

        reader.close()
        writer.close()

        # Stack into 4D arr, create Data obj, and save
        v_4D = np.stack(v_4D)

        data = Data(
            v=v_4D, frame_t=frame_t, events=events, 
            fps=fps, dense=False
        )
        data.save(f_out + '_shape.hdf5')
        data.visualize(f_out + '_viz-world_shape.mp4')
import cv2
import torch
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from ..detect import FAN
from ..recon import EMOCA
from ..recon.emoca.utils import tensor2image
from ..io import Data
from ..utils import get_logger

logger = get_logger()


def recon(video_path, events_path=None, cfg=None, device='cuda', out_dir=None,
          allow_pose=False):
    """ Reconstruction of all frames of a video. 
    
    Parameters
    ----------
    video_path : str, Path
        Path to video file to reconstruct
    events_path : str, Path
        Path to events file (a TSV or CSV file) containing
        info about experimental events; must have at least the
        columns 'onset' (in seconds) and 'trial_type'; optional
        columns are 'duration' and 'modulation'
    cfg : str
        Path to config file for reconstruction
    device : str
        Either "cuda" (for GPU) or "cpu"
    out_dir : str, Path
        Path to directory where recon data (and associated
        files) are saved
    allow_pose : bool
        If True, pose parameters are *not* set to 0 (so pose
        affects the vertex values)
    """

    if isinstance(video_path, str):
        video_path = Path(video_path)

    if not video_path.is_file():
        raise ValueError(f"File {video_path} does not exist!")

    logger.info(f'Starting 3D recon for {video_path}')
    
    if out_dir is None:
        out_dir = video_path.parent

    out_dir.mkdir(exist_ok=True, parents=True)

    # Init here, otherwise it's done every frame -> slow
    logger.info("Initializing FAN (detection) and EMOCA (recon)")
    fan = FAN(device=device)  # for face detection / cropping
    emoca = EMOCA(cfg=cfg, device=device)

    base_f = video_path.stem.replace('_video', '')
    reader = imageio.get_reader(video_path)
    sf = reader.get_meta_data()['fps']  # sf = sampling frequency

    # Use cv2 to get n_frames (not possible with imageio ...)
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if frame times (ft) and events (ev) files exist    
    ft_path = Path(str(video_path).replace('_video.mp4', '_frametimes.tsv'))
    if ft_path.is_file():
        # We're going to use the average number of frames
        # per second as FPS (and equivalently `sf`, sampling freq)
        frame_t = pd.read_csv(ft_path, sep='\t')['t'].to_numpy()
        sampling_period = np.diff(frame_t)
        sf, sf_std = 1 / sampling_period.mean(), 1 / sampling_period.std()
        logger.info(f"Average FPS/sampling frequency: {sf:.2f} (SD: {sf_std:.2f})")
    else:
       logger.warning(f"Did not find frame times file for {video_path} "
                      f"assuming constant FPS/sampling frequency ({sf})!")
       frame_t = np.linspace(0, n_frames * (1 / sf), endpoint=False, num=n_frames)

    # If `events_path` is not given, check if we can find it
    # in a file with the same name as the video, but different
    # identifier (_events.tsv)
    if events_path is None:
        # check for associated TSV path
        events_path = Path(str(video_path).replace('_video.mp4', '_events.tsv'))
  
    if events_path.is_file():
        events = pd.read_csv(events_path, sep='\t')
    else:
        logger.warning(f"Did not find events file for video {video_path}!")
        events = None

    # Init lists for reconstruced vertices across time (V) and
    # the framewise motion estimates (motion; global rot x/y/z, trans x/y, scale (z))
    v = []
    motion = []

    # For QC, we will write out the rendered shape on top of the original frame
    f_out = str(out_dir / base_f) + '_desc-recon'
    writer = imageio.get_writer(f_out + '_shape+img.gif', mode='I', fps=sf)
    
    # Loop across frames of video
    desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ] ')
    i = 0
    for frame in tqdm(reader, desc=f"{desc} Recon frames", total=n_frames):

        # Prepare frame to be used as background during rendering            
        orig_size = frame.shape[:2]
        frame_orig = (frame / 255.).transpose(2, 0, 1)  # channel, width, height
        frame_orig = torch.tensor(frame_orig).float()[None, ...].to('cuda')

        # Crop image and do forward pass on encoding stage
        frame_cropped = fan(frame)
        enc_dict = emoca.encode(frame_cropped)

        # Decode with `tform` params, so that the reconstructed vertices are in "image space"
        # which are subsequently rendered and passed to the writer; note that these operations
        # are for QC only and can thus be technically omitted
        dec_dict = emoca.decode(enc_dict, tform=fan.tform_params, orig_size=orig_size)
        rend_dict = emoca.render_dec(enc_dict, dec_dict, render_world=False,
                                    img_orig=frame_orig)

        img = tensor2image(rend_dict['shape_detail'][0], bgr=False)
        writer.append_data(img)
        
        # First, save motion parameters from the encoding stage
        if allow_pose:
            v.append(dec_dict['V'])
            continue

        motion.append(torch.cat((enc_dict['pose'][:, :3], enc_dict['cam']), axis=1))
        enc_dict['pose'][0, :3] = 0
        
        # Now, we decode again, but this time, we'll set the first three
        # pose params to 0 to get the reconstruction without any
        # rotation (also `tform` is set to `None` so everything is in
        # "world space")
        dec_dict = emoca.decode(enc_dict, tform=None)
        
        # We'll only save the `V` vertices, which are in "world space"
        # and thus not translated / scaled (so they do not contain
        # any global/rigid motion)
        v.append(dec_dict['V'])
        
        i += 1
        #if i > 20:
        #    break
        
    reader.close()
    writer.close()

    # Stack 2D arrs (nV x 3) into 3D arr (T x nV x 3) and
    # motion into 2D arr (nV x 6)
    v = torch.cat(v)  # tensor-format needed to render!

    if allow_pose:
        motion = None
    else:
        motion = torch.cat(motion).cpu().numpy()
        motion[:, :3] = np.degrees(motion[:, :3])  # rad. to degress

    if v.shape[0] < frame_t.size:
        # During debugging, sometimes there might be
        # fewer reconstructed frames than frame times
        frame_t = frame_t[:v.shape[0]]

    # Create Data object and save to disk as hdf5 file
    data = Data(v=v.cpu().numpy(), motion=motion, frame_t=frame_t,
                events=events, sf=sf, dense=False)

    data.save(f_out + '_shape.h5')
    data.render_video(f_out + '_shape.gif', emoca.render)
    data.plot_data(f_out + '_qc.png', n_pca=3)
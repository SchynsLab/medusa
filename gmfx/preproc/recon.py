import cv2
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from ..recon import EMOCA, FAN, Mediapipe
from ..io import MODEL2CLS
from ..utils import get_logger

logger = get_logger()


def videorecon(video_path, events_path=None, recon_model_name='mediapipe', cfg=None, device='cuda',
               out_dir=None, render_on_video=False):
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
    recon_model_name : str
        Name of reconstruction model, options are: 'emoca', 'emoca', 'mediapipe',
        'FAN-2D', and 'FAN-3D'
    cfg : str
        Path to config file for EMOCA reconstruction; ignored if not using emoca
    device : str
        Either "cuda" (for GPU) or "cpu" (ignored when using mediapipe)
    out_dir : str, Path
        Path to directory where recon data (and associated
        files) are saved; if `None`, same directory as video is used
    """

    if isinstance(video_path, str):
        video_path = Path(video_path)

    if not video_path.is_file():
        raise ValueError(f"File {video_path} does not exist!")

    logger.info(f'Starting recon using for {video_path}')
    
    if out_dir is None:
        out_dir = video_path.parent

    out_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Initializing {recon_model_name} recon model")
    if recon_model_name in ['emoca', 'emoca-dense']:
        fan = FAN(device=device)  # for face detection / cropping
        recon_model = EMOCA(cfg=cfg, device=device)
    elif recon_model_name == 'FAN-3D':
        recon_model = FAN(device=device, lm_type='3D')
    elif recon_model_name == 'mediapipe':
        recon_model = Mediapipe()
    else:
        raise NotImplementedError

    # Initialize video reader and extract some metadata
    reader = imageio.get_reader(video_path)
    sf = reader.get_meta_data()['fps']  # sf = sampling frequency
    frame_size = reader.get_meta_data()['size']
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
    recon_data = defaultdict(list)

    # For QC, we will write out the rendered shape on top of the original frame
    # (writer_recon) and the cropped image/landmarks/bounding box (writer_crop)
    base_f = video_path.stem.replace('_video', '')
    f_out = str(out_dir / base_f) + '_desc-recon'

    if recon_model_name in ['emoca', 'emoca-dense']:
        writer_crop = imageio.get_writer(f_out + '_crop.gif', mode='I', fps=sf)

    # Loop across frames of video
    desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ] ')
    i = 0
    for frame in tqdm(reader, desc=f"{desc} Recon frames", total=n_frames):
     
        if recon_model_name in ['emoca', 'emoca-dense']:

            # Crop image
            to_recon = fan.prepare_for_emoca(frame.copy())
            recon_data['tform'].append(fan.tform.params)
            # Save for visualization
            writer_crop.append_data(fan.viz_qc(return_rgba=True))
        else:
            to_recon = frame

        # Reconstruct and save whatever `recon_model`` returns in
        # `recon_data`
        out = recon_model(to_recon)
        for attr, data in out.items():
            recon_data[attr].append(data)
                
        #if i > 50:
        #    break
        i+=1

    reader.close()
    
    if recon_model_name in ['emoca']:
        writer_crop.close()

    # Concatenate all reconstuctions across time
    # such that the first dim represents time
    for attr, data in recon_data.items():
        recon_data[attr] = np.stack(data)

    T = recon_data['v'].shape[0]  # timepoints 
    if T < frame_t.size:
        # During debugging, sometimes there might be
        # fewer reconstructed frames than frame times
        frame_t = frame_t[:T]

    # Create Data object using the class corresponding to
    # the model (e.g., FlameData for `emoca`, MediapipeData for `mediapipe`)
    DataClass = MODEL2CLS[recon_model_name]
    data = DataClass(frame_t=frame_t, events=events, sf=sf, recon_model_name=recon_model_name,
                     image_size=frame_size, **recon_data)

    # Save data as hdf5 and visualize reconstruction 
    data.save(f_out + '_shape.h5')
    background = video_path if render_on_video else None
    data.plot_data(f_out + '_qc.png', plot_motion=True, n_pca=3)
    data.render_video(f_out + '_shape.gif', video=background)

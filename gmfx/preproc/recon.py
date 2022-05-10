import numpy as np
from collections import defaultdict

from ..recon import EMOCA, FAN, Mediapipe
from ..io import VideoData
from ..data import MODEL2CLS
from ..utils import get_logger

logger = get_logger()


def videorecon(video_path, events_path=None, recon_model_name='mediapipe', cfg=None, device='cuda',
               out_dir=None, render_on_video=False, render_crop=False, n_frames=None, scaling=None):
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
    render_on_video : bool
        Whether to render the reconstruction on top of the video;
        this may substantially increase rendering time!
    render_crop : bool
        Whether to render the cropping results (only relevant when using EMOCA,
        ignored otherwise)
    n_frames : int
        If not `None` (default), only reconstruct and render the first `n_frames`
        frames of the video; nice for debuggin
    """

    logger.info(f'Starting recon using for {video_path}')
    logger.info(f"Initializing {recon_model_name} recon model")
    
    # Initialize VideoData object here to use metadata
    # in recon_model (like img_size)
    video = VideoData(video_path, events=events_path)      

    # Initialize reconstruction model    
    if recon_model_name in ['emoca', 'emoca-dense']:
        fan = FAN(device=device)  # for face detection / cropping
        recon_model = EMOCA(cfg=cfg, device=device, img_size=video.img_size)
    elif recon_model_name == 'FAN-3D':
        recon_model = FAN(device=device, lm_type='3D')
    elif recon_model_name == 'mediapipe':
        recon_model = Mediapipe()
    else:
        raise NotImplementedError
    
    if out_dir is None:
        out_dir = video.path.parent
    
    out_dir.mkdir(exist_ok=True, parents=True)
    f_out = str(out_dir / str(video.path.name).replace('.mp4', '_desc-recon'))

    if recon_model_name in ['emoca'] and render_crop:
        video.create_writer(f_out, idf='crop', ext='gif')

    # Loop across frames of video, store results in `recon_data`
    recon_data = defaultdict(list)    
    for i, frame in video.loop(scaling=scaling):
     
        if recon_model_name in ['emoca']:

            # Crop image, add tform to emoca (for adding rigid motion
            # due to cropping), and add crop plot to writer
            frame = fan.prepare_for_emoca(frame)
            recon_model.tform = fan.tform.params
            
            if render_crop:
                video.write(fan.viz_qc(return_rgba=True))

        # Reconstruct and store whatever `recon_model`` returns
        # in `recon_data`
        out = recon_model(frame)
        for attr, data in out.items():
            recon_data[attr].append(data)
        
        if n_frames is not None:
            # If we only want to reconstruct a couple of
            # frames, stop if reached
            if i == n_frames:
                video.stop_loop()

    # Concatenate all reconstuctions across time
    # such that the first dim represents time
    for attr, data in recon_data.items():
        recon_data[attr] = np.stack(data)

    # Create Data object using the class corresponding to
    # the model (e.g., FlameData for `emoca`, MediapipeData for `mediapipe`)
    DataClass = MODEL2CLS[recon_model_name]
    kwargs = {**recon_data, **video.get_metadata()}
    data = DataClass(recon_model_name=recon_model_name, **kwargs)

    # Save data as hdf5 and visualize reconstruction 
    data.save(f_out + '_shape.h5')
    data.plot_data(f_out + '_qc.png', plot_motion=True, n_pca=3)
    background = video_path if render_on_video else None
    data.render_video(f_out + '_shape.gif', video=background, scaling=scaling)

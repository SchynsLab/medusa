import numpy as np
from pathlib import Path
from collections import defaultdict

from ..recon import EMOCA, FAN, Mediapipe
from ..io import VideoData
from ..core import MODEL2CLS
from ..utils import get_logger


def videorecon(
    video_path,
    events_path=None,
    recon_model_name="mediapipe",
    cfg=None,
    device="cuda",
    n_frames=None,
    loglevel='INFO'
):
    """Reconstruction of all frames of a video.

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
        Name of reconstruction model, options are: 'emoca', 'mediapipe',
        and 'FAN-3D'
    cfg : str
        Path to config file for EMOCA reconstruction; ignored if not using emoca
    device : str
        Either "cuda" (for GPU) or "cpu" (ignored when using mediapipe)
    n_frames : int
        If not `None` (default), only reconstruct and render the first `n_frames`
        frames of the video; nice for debugging
    loglevel : str
        Logging level, options are (in order of verbosity): 'DEBUG', 'INFO', 'WARNING',
        'ERROR', and 'CRITICAL'

    Returns
    -------
    data : medusa.core.*Data
        An object with a class inherited from ``medusa.core.BaseData``
        
    Examples
    --------
    Reconstruct a video using Mediapipe:
    
    >>> from medusa.data import get_example_video
    >>> vid = get_example_video()
    >>> data = videorecon(vid, recon_model_name='mediapipe')

    Reconstruct a video using FAN-3D, but only the first 50 frames of the video:

    >>> data = videorecon(vid, recon_model_name='FAN-3D', n_frames=50)
    >>> data.v.shape
    (50, 68, 3)
    """

    logger = get_logger()
    logger.setLevel(loglevel)
    logger.info(f"Starting recon using for {video_path}")
    logger.info(f"Initializing {recon_model_name} recon model")

    # Initialize VideoData object here to use metadata
    # in recon_model (like img_size)
    video = VideoData(video_path, events=events_path, loglevel=loglevel)

    # Initialize reconstruction model
    if recon_model_name in ["emoca", "emoca-dense"]:
        fan = FAN(device=device)  # for face detection / cropping
        recon_model = EMOCA(cfg=cfg, device=device, img_size=video.img_size)
    elif recon_model_name == "FAN-3D":
        recon_model = FAN(device=device, lm_type="3D")
    elif recon_model_name == "mediapipe":
        recon_model = Mediapipe()
    else:
        raise NotImplementedError

    # Loop across frames of video, store results in `recon_data`
    recon_data = defaultdict(list)
    for i, frame in video.loop():

        if recon_model_name in ["emoca"]:

            # Crop image, add tform to emoca (for adding rigid motion
            # due to cropping), and add crop plot to writer
            frame = fan.prepare_for_emoca(frame)
            recon_model.tform = fan.tform.params

        # Reconstruct and store whatever `recon_model`` returns
        # in `recon_data`
        out = recon_model(frame)
        for attr, data in out.items():
            recon_data[attr].append(data)

        if n_frames is not None:
            # If we only want to reconstruct a couple of
            # frames, stop if reached
            if i == (n_frames - 1):
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

    return data

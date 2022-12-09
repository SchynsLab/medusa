from collections import defaultdict

import numpy as np

from .. import DEVICE
from ..core import FLAME_MODELS, MODEL2CLS
from ..crop import LandmarkBboxCropModel
from ..io import VideoLoader
from ..log import get_logger
from . import Mediapipe
from .flame import DecaReconModel


def videorecon(video_path, recon_model="mediapipe", device=DEVICE, n_frames=None,
               batch_size=32, loglevel='INFO'):
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
    recon_model : str
        Name of reconstruction model, options are: 'emoca', 'mediapipe',
    device : str
        Either "cuda" (for GPU) or "cpu" (ignored when using mediapipe)
    n_frames : int
        If not `None` (default), only reconstruct and render the first `n_frames`
        frames of the video; nice for debugging
    batch_size : int
        Batch size (i.e., number of frames) processed by the reconstruction model
        in each iteration; decrease this number when you get out of memory errors
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
    >>> data = videorecon(vid, recon_model='mediapipe')
    """

    logger = get_logger(loglevel)
    logger.info(f"Starting recon using for {video_path}")
    logger.info(f"Initializing {recon_model} recon model")

    # Initialize VideoLoader object here to use metadata
    # in recon_model (like img_size)
    video = VideoLoader(video_path, batch_size=batch_size, loglevel=loglevel)
    metadata = video.get_metadata()

    # Initialize reconstruction model
    if recon_model in FLAME_MODELS:
        crop_model = LandmarkBboxCropModel(device=device)  # for face detection / cropping
        reconstructor = DecaReconModel(recon_model, device=device, img_size=metadata['img_size'])
    elif recon_model == "mediapipe":
        reconstructor = Mediapipe()
    else:
        raise NotImplementedError

    # Loop across frames of video, store results in `recon_data`
    recon_data = defaultdict(list)
    i_frame = 0

    for batch in video:
        inputs = {'imgs': batch}

        if recon_model in FLAME_MODELS:

            # Crop image, add tform to emoca (for adding rigid motion
            # due to cropping), and add crop plot to writer
            out_crop = crop_model(batch)
            inputs['imgs'] = out_crop.imgs_crop
            inputs['crop_mats'] = out_crop.crop_mats

        # Reconstruct and store whatever `recon_model`` returns
        # in `recon_data`
        outputs = reconstructor(**inputs)
        for attr, data in outputs.items():
            recon_data[attr].append(data)

        i_frame += outputs['v'].shape[0]
        if n_frames is not None:
            if i_frame >= n_frames:
                break

    # Mediapipe (and maybe future models) need to be closed in order to avoid
    # opening too many threads
    reconstructor.close()
    video.close()

    # Concatenate all reconstuctions across time
    # such that the first dim represents time
    for attr, data in recon_data.items():
        recon_data[attr] = np.concatenate(data, axis=0)
        if recon_data[attr].shape[0] != n_frames:
            recon_data[attr] = recon_data[attr][:n_frames, ...]

    DataClass = MODEL2CLS[recon_model]
    init_kwargs = {**recon_data, **video.get_metadata()}
    data = DataClass(recon_model=recon_model, tris=reconstructor.get_tris(),
                     **init_kwargs)

    return data

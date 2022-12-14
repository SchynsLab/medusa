from tqdm import tqdm

from .. import DEVICE
from ..containers import FLAME_MODELS
from ..crop import LandmarkBboxCropModel
from ..io import VideoLoader
from ..log import get_logger, tqdm_log
from . import Mediapipe
from .flame import DecaReconModel
from ..containers.results import BatchResults
from ..containers.fourD import Data4D


def videorecon(video_path, recon_model="mediapipe", device=DEVICE, n_frames=None,
               batch_size=32, loglevel='INFO'):
    """Reconstruction of all frames of a video.

    Parameters
    ----------
    video_path : str, Path
        Path to video file to reconstruct
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
    video = VideoLoader(video_path, batch_size=32, loglevel=loglevel)
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
    recon_results = BatchResults(device=device)
    i_frame = 0
    for batch in tqdm_log(video, logger, desc='Recon images'):
        inputs = {'imgs': batch}

        if recon_model in FLAME_MODELS:

            # Crop image, add tform to emoca (for adding rigid motion
            # due to cropping), and add crop plot to writer
            out_crop = crop_model(batch)
            del batch
            inputs['imgs'] = out_crop.pop('imgs_crop')
            inputs['crop_mats'] = out_crop.pop('crop_mats')
            recon_results.add(**out_crop)

        # Reconstruct and store whatever `recon_model`` returns
        # in `recon_data`
        outputs = reconstructor(**inputs)
        recon_results.add(**outputs)

        i_frame += outputs['v'].shape[0]
        if n_frames is not None:
            if i_frame >= n_frames:
                break

    # Mediapipe (and maybe future models) need to be closed in order to avoid
    # opening too many threads
    reconstructor.close()
    video.close()

    recon_results.concat(n_max=n_frames)
    recon_results.sort_faces(attr='v')

    metadata = video.get_metadata()
    tris = reconstructor.get_tris()
    init_kwargs = recon_results.to_dict(exclude=['lms', 'device', 'n_img', 'conf', 'bbox'])
    data = Data4D(video_metadata=metadata, tris=tris, **init_kwargs)

    return data

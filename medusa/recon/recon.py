from ..defaults import DEVICE, FLAME_MODELS
from ..containers.fourD import Data4D
from ..containers.results import BatchResults
from ..crop import LandmarkBboxCropModel
from ..io import VideoLoader
from ..log import get_logger, tqdm_log
from .flame import DecaReconModel
from .mpipe import Mediapipe


def videorecon(
    video_path,
    recon_model="mediapipe",
    device=DEVICE,
    n_frames=None,
    batch_size=32,
    loglevel="INFO",
    **kwargs,
):
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
    video = VideoLoader(
        video_path, batch_size=batch_size, loglevel=loglevel, device=device
    )
    metadata = video.get_metadata()

    # Initialize reconstruction model
    if recon_model in FLAME_MODELS:
        crop_model = LandmarkBboxCropModel(
            device=device
        )  # for face detection / cropping
        reconstructor = DecaReconModel(
            recon_model, device=device, img_size=metadata["img_size"], **kwargs
        )
    elif recon_model == "mediapipe":
        reconstructor = Mediapipe(static_image_mode=False, device=device, **kwargs)
    else:
        raise NotImplementedError

    # Loop across frames of video, store results in `recon_data`
    recon_results = BatchResults(device=device)
    for batch in tqdm_log(video, logger, desc="Recon images"):
        inputs = {"imgs": batch}

        if recon_model in FLAME_MODELS:

            out_crop = crop_model(batch)
            del batch
            inputs["imgs"] = out_crop.pop("imgs_crop")
            inputs["crop_mats"] = out_crop.pop("crop_mats")
            recon_results.add(**out_crop)

        # Reconstruct and store whatever `recon_model`` returns
        # in `recon_data`
        if inputs["imgs"] is not None:
            outputs = reconstructor(**inputs)
            recon_results.add(**outputs)

        if n_frames is not None:
            if recon_results.n_img >= n_frames:
                break

    recon_results.concat(n_max=n_frames)
    recon_results.sort_faces(attr="v")
    recon_results.filter_faces(present_threshold=0.1)

    if getattr(recon_results, "v", None) is None:
        raise ValueError("No faces in entire video!")

    metadata = video.get_metadata()
    tris = reconstructor.get_tris()
    cam_mat = reconstructor.get_cam_mat()
    init_kwargs = recon_results.to_dict(
        exclude=["lms", "device", "n_img", "conf", "bbox"]
    )
    data = Data4D(video_metadata=metadata, tris=tris, cam_mat=cam_mat, **init_kwargs)

    # Mediapipe (and maybe future models) need to be closed in order to avoid
    # opening too many threads
    reconstructor.close()
    video.close()

    return data

""" This module contains functions to load in example data, which
is used for examples and tests. The example data is the following video:
https://www.pexels.com/video/close-up-of-a-woman-showing-different-facial-expressions-3063839/
made freely available by Wolfgang Langer.

The video was trimmed to 10 seconds and resized in order to reduce disk space.
"""

import cv2
import torch
from pathlib import Path

from ..io import VideoLoader


def get_example_frame(load_numpy=False, load_torch=False, device='cuda'):
    """Loads an example frame from the example video.
    
    Parameters
    ----------
    load_numpy : bool
        Whether to load it as a numpy array
    load_torch : bool
        Whether to load it as a torch array
    device : str
        Either 'cuda' or 'cpu'; ignored when ``load_torch`` is False

    Returns
    -------
    img : pathlib.Path, np.ndarray, torch.Tensor
        A path or a 3D numpy array/torch Tensor of shape frame width x height x 3 (RGB)

    Notes
    -----
    If both ``load_numpy`` and ``load_torch`` are False, then just
    a ``pathlib.Path`` object is returned.

    Examples
    --------
    >>> # Load path to example image frame
    >>> img = get_example_frame()
    >>> img.is_file()
    True
    >>> # Load file as numpy array
    >>> img = get_example_frame(load_numpy=True)
    >>> img.shape
    (384, 480, 3)
    """

    if load_numpy and load_torch:
        raise ValueError("Set either 'load_numpy' or 'load_torch' to True, not both!")

    here = Path(__file__).parent
    img_path = here / "example_data/example_frame.png"
    
    if not load_torch and not load_numpy:
        return img_path
        
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if load_torch:
        img = torch.from_numpy(img).to(device)

    return img


def get_example_video(return_videoloader=False, **kwargs):
    """Retrieves the path to an example video file.

    Parameters
    ----------
    return_videoloader : bool
        Returns the video as a ``VideoLoader`` object
    kwargs : dict
        Extra parameters passed to the ``VideoLoader`` initialization;
        ignored when ``return_videoloader`` is False

    Returns
    -------
    path : pathlib.Path, VideoLoader
        A Path object pointing towards the example
        video file or a ``VideoLoader`` object

    Examples
    --------
    Get just the file path (as a ``pathlib.Path`` object)

    >>> path = get_example_video()
    >>> path.is_file()
    True

    Get it as a ``VideoLoader`` object to quickly get batches of images already
    loaded on and formatted for GPU:

    >>> vid = get_example_video(return_videoloader=True, batch_size=32)
    >>> # We can loop over `vid` or just get a single batch, as below:
    >>> img_batch = next(vid)
    >>> img_batch.shape
    torch.Size([32, 384, 480, 3])
    """

    here = Path(__file__).parent
    vid = here / "example_data/example_vid.mp4"

    if return_videoloader:
        vid = VideoLoader(vid, **kwargs)

    return vid


def get_example_h5(load=False, model="mediapipe", as_path=True):
    """Retrieves an example hdf5 file with reconstructed 4D
    data from the example video.

    Parameters
    ----------
    load : bool
        Whether to return the hdf5 file loaded in memory (``True``)
        or to just return the path to the file
    model : str
        Model used to reconstruct the data; either 'mediapipe' or
        'emoca'
    as_path : bool
        Whether to return the path as a ``pathlib.Path`` object (``True``)
        or just a string (``False``); ignored when ``load`` is ``True``

    Returns
    -------
    MediapipeData, FlameData, str, Path
        When ``load`` is ``True``, returns either a ``MediapipeData``
        or a ``FlameData`` object, otherwise a string or ``pathlib.Path``
        object to the file

    Examples
    --------
    >>> path = get_example_h5(load=False, as_path=True)
    >>> path.is_file()
    True

    # Get hdf5 file already loaded in memory
    >>> data = get_example_h5(load=True, model='mediapipe')
    >>> data.recon_model
    'mediapipe'
    >>> data.v.shape  # check out reconstructed vertices
    (232, 468, 3)
    """
    from ..io import load_h5

    here = Path(__file__).parent
    path = here / f"example_data/example_vid_{model}.h5"

    if not as_path:
        path = str(path)

    if load:
        return load_h5(path)
    else:
        return path

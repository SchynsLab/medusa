"""This module contains functions to load in example data, which is used for
examples and tests. The example data is the following video:
https://www.pexels.com/video/close-up-of-a-woman-showing-different-facial-
expressions-3063839/ made freely available by Wolfgang Langer.

The video was trimmed to 10 seconds and resized in order to reduce disk
space.
"""

from pathlib import Path

import torch
from torchvision.io import read_image

from ..recon import videorecon
from ..defaults import DEVICE
from ..containers import Data4D
from ..io import VideoLoader


def get_example_image(n_faces=None, load=True, device=DEVICE, channels_last=False,
                      dtype=torch.float32):
    """Loads an example frame from the example video.

    Parameters
    ----------
    n_faces : int, list, None
        If None, it will return the default (example) image (the first frame from
        the example video); if an integer, it will return an image with that many
        faces in it (see medusa/data/example_data/images folder); if a list (or tuple),
        it will return a list of images with the number of faces specified in the list
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
    >>> img = get_example_image()
    >>> img.is_file()
    True
    >>> # Load file as numpy array
    >>> img = get_example_image(load_numpy=True)
    >>> img.shape
    (384, 480, 3)
    """

    data_dir = Path(__file__).parent / "example_data/images"

    if n_faces is None:
        img_path = [data_dir / 'example_frame.png']
    else:
        if isinstance(n_faces, (list, tuple)):
            img_path = [data_dir / f'{nf}_face.jpg' for nf in n_faces]
        else:
            img_path = [data_dir / f'{n_faces}_face.jpg']

    imgs = []
    for f in img_path:
        if not f.is_file():
            raise FileNotFoundError(f"Could not find example image file {f}!")

        if not load:
            imgs.append(f)
        else:
            img = read_image(str(f)).to(device)
            if channels_last:
                img = img.permute(1, 2, 0)

            imgs.append(img)

    if load:
        if len(imgs) == 1:
            imgs = imgs[0].unsqueeze(0)
        else:
            imgs = torch.stack(imgs)

        imgs = imgs.to(dtype)

    return imgs


def get_example_video(n_faces=None, return_videoloader=False, **kwargs):
    """Retrieves the path to an example video file.

    Parameters
    ----------
    n_faces : int, None
        If None, it will return the default (example) video; if an integer, it will
        return an image with that many faces in it (see medusa/data/example_data/videos folder)
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

    data_dir = Path(__file__).parent / "example_data/videos"

    if n_faces is None:
        f_name = 'example_vid.mp4'
    else:
        f_name = f'{n_faces}_face.mp4'

    vid = data_dir / f_name

    if not vid.is_file():
        raise FileNotFoundError(f"Could not find video {vid}")

    if return_videoloader:
        vid = VideoLoader(vid, **kwargs)

    return vid


def get_example_data4d(n_faces=None, load=False, model="mediapipe", device=DEVICE):
    """Retrieves an example hdf5 file with reconstructed 4D data from the
    example video.

    Parameters
    ----------
    n_faces : int, None
        If None, it will return the reconstruction from the default (example) video; if
        an integer, it will return the recon data from the video with that many faces in
        it (see medusa/data/example_data/videos folder)
    load : bool
        Whether to return the hdf5 file loaded in memory (``True``)
        or to just return the path to the file
    model : str
        Model used to reconstruct the data; either 'mediapipe' or
        'emoca'

    Returns
    -------
    MediapipeData, FlameData, str, Path
        When ``load`` is ``True``, returns either a ``MediapipeData``
        or a ``FlameData`` object, otherwise a string or ``pathlib.Path``
        object to the file

    Examples
    --------
    >>> path = get_example_data4d(load=False, as_path=True)
    >>> path.is_file()
    True

    # Get hdf5 file already loaded in memory
    >>> data = get_example_data4d(load=True, model='mediapipe')
    >>> data.recon_model
    'mediapipe'
    >>> data.v.shape  # check out reconstructed vertices
    (232, 468, 3)
    """

    data_dir = Path(__file__).parent / "example_data/recons"

    if n_faces is None:
        f_name = f'example_vid_{model}.h5'
    else:
        f_name = f'{n_faces}_face_{model}.h5'

    recon = data_dir / f_name

    if not recon.is_file():
        vid = get_example_video(n_faces)
        data_4d = videorecon(vid, recon_model=model, device=device, loglevel="ERROR")
        data_4d.save(recon)
        if load:
            return data_4d
        else:
            return recon

    if load:
        return Data4D.load(recon, device=device)
    else:
        return recon

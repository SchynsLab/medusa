"""Module with functionality (mostly) for working with video data.

The ``VideoLoader`` class allows for easy looping over frames of a video
file, which is used in the reconstruction process (e.g., in the
``videorecon`` function).
"""

from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, IterableDataset
from trimesh import Trimesh

from . import DEVICE
from .log import get_logger


class VideoLoader(DataLoader):
    """" Contains (meta)data and functionality associated with video files (mp4
    files only currently).

    Parameters
    ----------
    path : str, Path
        Path to mp4 file
    rescale_factor : float
        Rescale factor of video frames (e.g., 0.25 means scale each dimension to 25% of original);
        if ``None`` (default), the image is not resized
    n_preload : int
        Number of video frames to preload before batching
    loglevel : str
        Logging level (e.g., 'INFO' or 'WARNING')

    Raises
    ------
    ValueError
        If `n_preload` is not a multiple of `batch_size`
    """

    def __init__(self, path, batch_size=32, device=DEVICE,
                 channels_first=False, loglevel='INFO', **kwargs):

        self.logger = get_logger(loglevel)
        self.channels_first = channels_first
        self.device = device
        self._validate(path)
        dataset = VideoDataset(path, device)

        super().__init__(dataset, batch_size, num_workers=0, pin_memory=True, **kwargs)
        self._metadata = self._extract_metadata()

    def get_metadata(self):
        """Returns all (meta)data needed for initialization of a Data
        object."""

        return self._metadata

    def close(self):
        """Closes the opencv videoloader in the underlying pytorch Dataset."""
        self.dataset.close()

    def _validate(self, path):
        """Validates some of the init arguments."""

        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_file():
            raise ValueError(f"File {path} does not exist!")

    def _extract_metadata(self):
        """Extracts some metadata from Dataset and exposes it to the loader."""

        tmp = self.dataset.metadata
        end = tmp['n'] / tmp['fps']
        #frame_t = np.linspace(0, end, endpoint=False, num=tmp['n'])

        return {
            #'frame_t': frame_t,
            'img_size': tmp['size'],
            'fps': tmp['fps']
        }

    #def __len__(self):
    #    """Utility function to easily access number of video frames."""
    #    return len(self.dataset)

    def __iter__(self):
        """Little hack to put data on device automatically."""
        for batch in super().__iter__():
            batch = batch.to(self.device, non_blocking=True)
            if self.channels_first:
                batch = batch.permute(0, 3, 1, 2)

            yield batch


class VideoDataset(IterableDataset):
    """A pytorch Dataset class based on loading frames from a single video.

    Parameters
    ----------
    video : pathlib.Path, str
        A video file (any format that cv2 can handle)
    rescale_factor : float
        Factor with which to rescale the input image (for speed)
    n_preload : int
        How many frames to preload before batching; higher values will
        take up more RAM, but result in faster loading
    device : str
        Either 'cuda' (for GPU) or 'cpu'
    """
    def __init__(self, video, device='cuda'):

        self.device = device
        self._container = av.open(str(video), mode='r')
        self._container.streams.video[0].thread_type = 'AUTO'#thread_type
        self._reader = self._container.decode(video=0)
        self.metadata = self._get_metadata()

    def _get_metadata(self):

        stream = self._container.streams.video[0]
        fps = int(stream.average_rate)
        n = stream.frames
        w = stream.codec_context.width
        h = stream.codec_context.height

        #if self.rescale_factor is not None:
        #    w = round(self.rescale_factor * w)
        #    h = round(self.rescale_factor * h)

        return {
            'size': (w, h),
            'n': n,
            'fps': fps
        }

    def __len__(self):

        return self.metadata['n']

    def __iter__(self):
        for img in self._reader:
            yield img.to_ndarray(format='rgb24')

    def close(self):
        """Closes the cv2 videoreader and free up memory."""
        self._container.close()


class VideoWriter:
    """A PyAV based images-to-video writer.

    Parameters
    ----------
    path : str, Path
        Output path (including extension)
    fps : float, int
        Frames per second of output video; if float, it's rounded
        and cast to int
    codec : str
        Video codec to use (e.g., 'mpeg4', 'libx264', 'h264')
    pix_fmt : str
        Pixel format; should be compatible with codec
    """
    def __init__(self, path, fps, codec='libx264', pix_fmt='yuv420p'):

        self._container = av.open(str(path), mode='w')
        self._stream = self._container.add_stream(codec, int(round(fps)))
        self._stream.pix_fmt = pix_fmt

    def write(self, imgs):
        """Writes one or more images to the video stream.

        Parameters
        ----------
        imgs : array_like
            A torch tensor or numpy array with image data; can be
            a single image or batch of images
        """
        if imgs.ndim == 3:
            imgs = imgs[None, ...]

        if imgs.shape[1] in (3, 4):
            imgs = imgs.permute(0, 2, 3, 1)

        if torch.is_tensor(imgs):
            imgs = imgs.cpu().numpy()

        imgs = imgs.astype(np.uint8)

        b, h, w, c = imgs.shape
        if self._stream.width is None:
            self._stream.width = w
            self._stream.height = h

        for i in range(b):
            img = imgs[i]
            img = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in self._stream.encode(img):
                self._container.mux(packet)

    def close(self):
        """Closes the video stream."""
        for packet in self._stream.encode():
            self._container.mux(packet)

        self._container.close()


def load_inputs(inputs, load_as='torch', channels_first=True,
                with_batch_dim=True, dtype='float32', device=DEVICE):
    """Generic image loader function, which also performs some basic
    preprocessing and checks. Is used internally for crop models and
    reconstruction models.

    Parameters
    ----------
    inputs : str, Path, iterable, array_like
        String or Path to a single image or an iterable (list, tuple) with
        multiple image paths, or a numpy array or torch Tensor with already
        loaded images
    load_as : str
        Either 'torch' (returns torch Tensor) or 'numpy' (returns numpy ndarray)
    to_bgr : bool
        Whether the color channel is ordered BGR (True) or RGB (False); only
        works when inputs are image path(s)
    channels_first : bool
        Whether the data is ordered as (batch_size, 3, h, w) (True) or
        (batch_size, h, w, 3) (False)
    with_batch_dim : bool
        Whether a singleton batch dimension should be added if there's only
        a single image
    dtype : str
        Data type to be used for loaded images (e.g., 'float32', 'float64', 'uint8')
    device : str
        Either 'cuda' (for GPU) or 'cpu'; ignored when ``load_as='numpy'``

    Returns
    -------
    imgs : np.ndarray, torch.Tensor
        Images loaded in memory; object depends on the ``load_as`` parameter

    Examples
    --------
    Load a single image as a torch Tensor:
    >>> from medusa.data import get_example_frame
    >>> path = get_example_frame()
    >>> img = load_inputs(path, device='cpu')
    >>> img.shape
    torch.Size([1, 3, 384, 480])

    Or as a numpy array (without batch dimension):

    >>> img = load_inputs(path, load_as='numpy', with_batch_dim=False)
    >>> img.shape
    (3, 384, 480)

    Putting the channel dimension last:

    >>> img = load_inputs(path, load_as='numpy', channels_first=False)
    >>> img.shape
    (1, 384, 480, 3)

    Setting the data type to uint8 instead of float32:

    >>> img = load_inputs(path, load_as='torch', dtype='uint8', device='cpu')
    >>> img.dtype
    torch.uint8

    Loading in a list of images:

    >>> img = load_inputs([path, path], load_as='numpy')
    >>> img.shape
    (2, 3, 384, 480)
    """

    if not load_as in ('torch', 'numpy'):
        raise ValueError("'load_as' should be either 'torch' or 'numpy'!")

    if not device in ('cuda', 'cpu'):
        raise ValueError("'device' should be either 'cuda' or 'cpu'!")

    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    if isinstance(inputs, (list, tuple)):
        imgs = []
        for inp in inputs:

            if isinstance(inp, (np.ndarray, torch.Tensor)):
                if inp.ndim == 4 and inp.shape[0] == 1:
                    inp = inp.squeeze()

                imgs.append(inp)
                continue

            if not isinstance(inp, Path):
                inp = Path(inp)

            if not inp.is_file():
                raise ValueError(f"Input '{inp}' does not exist!")

            img = cv2.imread(str(inp))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            imgs.append(img)

        if torch.is_tensor(imgs[0]):
            imgs = torch.stack(imgs, dim=0).to(device)
        else:
            imgs = np.stack(imgs)
    else:
        # If already torch or numpy, do nothing
        imgs = inputs

    if load_as == 'torch' and isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs.copy()).to(device)
    elif load_as == 'numpy' and isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()

    if imgs.ndim == 3:
        # Adding batch dim for now
        imgs = imgs[None, ...]

    if imgs.shape[1] != 3 and channels_first:
        if isinstance(imgs, np.ndarray):
            imgs = imgs.transpose(0, 3, 1, 2)
        else:  # assume torch
            imgs = imgs.permute(0, 3, 1, 2)

    if imgs.shape[1] == 3 and not channels_first:
        if isinstance(imgs, np.ndarray):
            imgs = imgs.transpose(0, 2, 3, 1)
        else:  # assume torch
            imgs = imgs.permute(0, 2, 3, 1)

    if isinstance(imgs, np.ndarray):
        imgs = imgs.astype(getattr(np, dtype))
    else:
        imgs = imgs.to(dtype=getattr(torch, dtype))

    if not with_batch_dim:
        imgs = imgs.squeeze()

    return imgs


def save_obj(v, f, f_out):
    if not isinstance(f_out, Path):
        f_out = Path(f_out)

    if not f_out.suffix == '.obj':
        raise ValueError("Filename should end in .obj!")

    mesh = Trimesh(v, f)
    mesh.export(f_out)


def download_file(url, f_out, data=None, verify=True, overwrite=False, cmd_type='post'):
    if f_out.is_file() and not overwrite:
        return

    with getattr(requests, cmd_type)(url, stream=True, verify=True, data=data) as r:
        r.raise_for_status()
        with open(f_out, 'wb') as f:
            f.write(r.content)

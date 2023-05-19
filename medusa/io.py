"""Module with functionality (mostly) for working with video data.

The ``VideoLoader`` class allows for easy looping over frames of a video
file, which is used in the reconstruction process (e.g., in the
``videorecon`` function).
"""

from pathlib import Path

import av
import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, Dataset, Subset
from torchvision.transforms.functional import crop

from .defaults import DEVICE


class VideoLoader(DataLoader):
    """Contains (meta)data and functionality associated with video files (mp4
    files only currently).

    Parameters
    ----------
    video_path : str, Path
        Path to mp4 file
    dataset_type : str
        One of 'iterable', 'map', or 'subset'. If 'iterable', batches are loaded
        sequentially from the video file. If 'map', frames can be loaded in any order
        and 'subset' allows for loading a subset of frames (using the 'frames' arg)
    batch_size : int
        Batch size to use when loading frames
    device : str
        Either 'cpu' or 'cuda'
    frames : list
        List of frame indices to loaded; only relevant when dataset_type is 'subset'
    **kwargs
        Extra keyword arguments passed to the initialization of the parent class
    """

    def __init__(self, video_path, dataset_type='iterable', batch_size=32, device=DEVICE,
                 frames=None, **kwargs):
        """Initializes a VideoLoader object."""

        self.video_path = video_path
        self.device = device
        self.crop = crop
        self._validate(video_path)

        if dataset_type == 'iterable':
            dataset = VideoIterableDataset(video_path)
        elif dataset_type in ['map', 'subset']:
            dataset = VideoMapDataset(video_path)
        else:
            raise ValueError("dataset_type must be one of 'iterable', 'map', or 'subset'")

        self._metadata = self._extract_metadata(dataset)

        if dataset_type == 'subset':
            dataset = Subset(dataset, frames)

        pin_memory = True if device == 'cuda' else False
        super().__init__(dataset, batch_size, num_workers=0, pin_memory=pin_memory,
                         *kwargs)

    def get_metadata(self):
        """Returns all (meta)data needed for initialization of a Data object.

        Returns
        -------
        A dictionary with keys "img_size" (image size of frames), "n_img" (total number
        of frames), and "fps" (frames-per-second)
        """

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

    def _extract_metadata(self, dataset):
        """Extracts some metadata from Dataset and exposes it to the loader."""
        return dataset.metadata

    def crop(self, batch, crop_params):
        """Crops an image batch.

        Parameters
        ----------
        batch : torch.tensor
            A B x H x W x 3 tensor with image data
        crop_params : tuple
            A tuple of (x0, y0, x1, y1) crop parameters
        """

        self.metadata["img_size"] = (crop_params[2], crop_params[3])
        return crop(batch.permute(0, 3, 1, 2), *crop_params).permute(0, 2, 3, 1)


class _VideoDatasetMixin:
    """A mixin class for VideoIterableDataset and VideoMapDataset.

    Parameters
    ----------
    video_path : Path, str
        Path to a video file
    """
    def __init__(self, video_path):
        """Initializes a VideoDataset object."""

        self.video_path = video_path
        self._container = av.open(str(video_path), mode="r")
        self._container.streams.video[0].thread_type = "AUTO"  # thread_type
        self._reader = self._container.decode(video=0)
        self.metadata = self._get_metadata()

    def _get_metadata(self):
        """Extracts some metadata from the stream."""
        stream = self._container.streams.video[0]
        fps = int(stream.average_rate)
        n = stream.frames
        w = stream.codec_context.width
        h = stream.codec_context.height

        return {"img_size": (w, h), "n_img": n, "fps": fps}

    def __len__(self):
        """Returns the number of frames in the video."""
        return self.metadata["n_img"]

    def close(self):
        """Closes the pyav videoreader and free up memory."""
        self._container.close()


class VideoIterableDataset(IterableDataset, _VideoDatasetMixin):
    """A pytorch Dataset class based on loading frames from a single video.

    Parameters
    ----------
    video_path : Path, str
        A video file (any format that pyav can handle)
    device : str
        Either 'cuda' (for GPU) or 'cpu'
    """

    def __init__(self, video_path):
        """Initializes a VideoDataset object."""
        IterableDataset.__init__(self)
        _VideoDatasetMixin.__init__(self, video_path)

    def __iter__(self):
        """Overrides parent method to make sure each image is a numpy array.

        Note to self: do not cast to torch here, because doing this
        later is way faster.
        """
        for img in self._reader:
            # CxHxW format
            yield img.to_ndarray(format="rgb24").transpose(2, 0, 1).astype(np.float32)


class VideoMapDataset(Dataset, _VideoDatasetMixin):
    """A pytorch Dataset class based on loading frames from a single video.

    Parameters
    ----------
    video_path : Path, str
        A video file (any format that pyav can handle)
    device : str
        Either 'cuda' (for GPU) or 'cpu'
    """

    def __init__(self, video_path):
        """Initializes a VideoDataset object."""
        Dataset.__init__(self)
        _VideoDatasetMixin.__init__(self, video_path)
        self.counter = 0

    def __getitem__(self, idx):

        if idx < self.counter:
            raise ValueError(f"Indices ({idx}) passed to VideoMapDataset should be consecutive!")

        for img in self._reader:
            self.counter += 1
            if (self.counter - 1) == idx:
                return img.to_ndarray(format="rgb24").transpose(2, 0, 1).astype(np.float32)
        else:
            raise ValueError(f"Something went wrong for index {idx}!")


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
    size : tuple[int]
        Desired output size of video (if ``None``, wil be set the first time a frame
        is written)
    """

    def __init__(self, path, fps, codec="libx264", pix_fmt="yuv420p", size=None):
        """Initializes a VideoWriter object."""
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream(codec, int(round(fps)))
        self._stream.pix_fmt = pix_fmt
        self.size = size

        if size is not None:
            self._stream.width = size[0]
            self._stream.height = size[1]

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

        b, h, w, _ = imgs.shape
        if self.size is None:
            self._stream.width = w
            self._stream.height = h

        for i in range(b):
            img = imgs[i]
            img = av.VideoFrame.from_ndarray(img, format="rgb24")

            for packet in self._stream.encode(img):
                self._container.mux(packet)

    def close(self):
        """Closes the video stream."""
        for packet in self._stream.encode():
            self._container.mux(packet)

        self._container.close()


def load_inputs(inputs, load_as="torch", channels_first=True, with_batch_dim=True,
                dtype="float32", device=DEVICE):
    """Generic image loader function, which also performs some basic
    preprocessing and checks. Is used internally for detection, crop, and
    reconstruction models.

    Parameters
    ----------
    inputs : str, Path, iterable, array_like
        String or ``Path`` to a single image or an iterable (list, tuple) with
        multiple image paths, or a numpy array or torch Tensor with already
        loaded images (in which the first dimension represents the number of images)
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
    imgs : np.ndarray, torch.tensor
        Images loaded in memory; object depends on the ``load_as`` parameter

    Examples
    --------
    Load a single image as a torch Tensor:
    >>> from medusa.data import get_example_image
    >>> path = get_example_image()
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

    if not load_as in ("torch", "numpy"):
        raise ValueError("'load_as' should be either 'torch' or 'numpy'!")

    if not device in ("cuda", "cpu"):
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

            img = np.array(Image.open(str(inp)))
            imgs.append(img)

        if torch.is_tensor(imgs[0]):
            imgs = torch.stack(imgs, dim=0).to(device)
        else:
            imgs = np.stack(imgs)
    else:
        # If already torch or numpy, do nothing
        imgs = inputs

    if load_as == "torch" and isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs.copy()).to(device)
    elif load_as == "numpy" and isinstance(imgs, torch.Tensor):
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


def download_file(url, f_out, data=None, verify=True, overwrite=False, cmd_type="post"):
    """Downloads a file using requests. Used internally to download external
    data.

    Parameters
    ----------
    url : str
        URL of file to download
    f_out : Path
        Where to save the downloaded file
    data : dict
        Extra data to pass to post request
    verify : bool
        Whether to verify the request
    overwrite : bool
        Whether to overwrite the file when it already exists
    cmd_type : str
        Either 'get' or 'post'
    """
    if f_out.is_file() and not overwrite:
        return

    with getattr(requests, cmd_type)(url, stream=True, verify=verify, data=data) as r:
        r.raise_for_status()
        with open(f_out, "wb") as f:
            f.write(r.content)


def load_obj(f, device=None):
    """Loads data from obj file, based on the DECA implementation, which in
    turn is based on the pytorch3d implementation.

    Parameters
    ----------
    f : str, Path
        Filename of object file
    device : str, None
        If None, returns numpy arrays. Otherwise, returns torch tensors on this device


    Returns
    -------
    out : dict
        Dictionary with outputs (keys: 'v', 'tris', 'vt', 'tris_uv')
    """

    with open(f, 'r') as f_in:
        lines = [line.strip() for line in f_in]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int64).reshape((-1, 3)) - 1
    uvcoords = np.array(uvcoords, dtype=np.float32)
    uv_faces = np.array(uv_faces, dtype=np.int64).reshape((-1, 3)) - 1

    if device is not None:
        verts = torch.as_tensor(verts, dtype=torch.float32, device=device)
        uvcoords = torch.as_tensor(uvcoords, dtype=torch.float32, device=device)
        faces = torch.as_tensor(faces, dtype=torch.long, device=device)
        uv_faces = torch.as_tensor(uv_faces, dtype=torch.long, device=device)

    out = {'v': verts, 'tris': faces, 'vt': uvcoords, 'tris_uv': uv_faces}
    return out


def save_obj(f, data):
    """Saves data to an obj file, based on the implementation from PRNet.

    Parameters
    ----------
    f : str, Path
        Path to save file to
    data : dict
        Dictionary with 3D mesh data, with keys 'v', 'tris', and optionally 'vt'
    """

    for k in data.keys():
        if torch.is_tensor(data[k]):
            data[k] = data[k].cpu().numpy().copy()

    v = data.get('v')
    tris = data.get('tris') + 1
    vt = data.get('vt', None)

    # write obj
    with open(f, 'w') as f_out:

        for i in range(v.shape[0]):
            f_out.write(f"v {v[i, 0]} {v[i, 1]} {v[i, 2]} \n")

        if vt is not None:
            for i in range(vt.shape[0]):
                f_out.write(f"vt {vt[i, 0]} {vt[i, 1]}\n")

        for i in range(tris.shape[0]):
            f_out.write(f"f {tris[i, 0]} {tris[i, 1]} {tris[i, 2]}\n")

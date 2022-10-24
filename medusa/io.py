""" Module with functionality (mostly) for working with video data. 
The `VideoData` class allows for easy looping over frames of a video file,
which is used in the reconstruction process (e.g., in the ``videorecon`` function).
"""

import cv2
import h5py
import logging
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from skimage.transform import rescale

from .utils import get_logger


class VideoData:
    """ " Contains (meta)data and functionality associated
    with video files (mp4 files only currently).

    Parameters
    ----------
    path : str, Path
        Path to mp4 file
    events : str, Path
        Path to a TSV file with event information (optional);
        should contain at least the columns 'onset' and 'trial_type'
    scaling : float
        Scaling factor of video frames (e.g., 0.25 means scale to 25% of original)
    loglevel : str  
        Logging level (e.g., 'INFO' or 'WARNING')

    Attributes
    ----------
    sf : int
        Sampling frequency (= frames per second, fps) of video
    n_img : int
        Number of images (frames) in the video
    img_size : tuple
        Width and height (in pixels) of the video
    frame_t : np.ndarray
        An array of length `self.n_img` with the onset of each
        frame of the video
    """

    def __init__(self, path, events=None, find_files=True, scaling=None, loglevel='INFO'):
        self.path = Path(path)
        self.events = events if events is None else Path(events) 
        self.scaling = scaling
        self.logger = get_logger(loglevel)
        self._validate()
        self._extract_metadata()

        if find_files:
            self._find_events()
            self._find_frame_t()

    def _validate(self):
        if not self.path.is_file():
            raise ValueError(f"File {self.path} does not exist!")

        sfx = self.path.suffix
        if sfx not in [".mp4", ".gif", ".avi"]:
            # Other formats might actually work, but haven't been tested yet
            raise ValueError(f"Only mp4/gif videos are supported, not {sfx[1:]}!")

    def _extract_metadata(self):
        """Extracts some metadata from the video needed for preprocessing
        functionality later on."""

        cap = cv2.VideoCapture(str(self.path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.sf = float(cap.get(cv2.CAP_PROP_FPS))
        self.img_size = (w, h)
        self.n_img = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

    def _find_events(self):
        """If not already supplied upon initialization of the Video object,
        this method trieds to find an event file associated with the video,
        which is assumed to have the same name as the video but ending in
        "_events.tsv" instead of ".mp4". If it's not found, it is simply
        ignored."""

        if self.events is None:
            # check for associated TSV path
            self.events = Path(str(self.path).replace(self.path.suffix, "_events.tsv"))

        if self.events.is_file():
            self.events = pd.read_csv(self.events, sep="\t")
        else:
            self.events = None

    def _find_frame_t(self):
        """Looks for a "frame times" file associated with the video,
        which is assumed to have the same name as the video but ending
        in "_frametimes.tsv" instead of ".mp4". If it's not found,
        the frame times are assumed to be constant and a multiple of
        the fps (frames per second)."""

        # Check if frame times (ft) and events (ev) files exist
        ft_path = Path(str(self.path).replace(self.path.suffix, "_frametimes.tsv"))
        if ft_path.is_file():
            # We're going to use the average number of frames
            # per second as FPS (and equivalently `sf`, sampling freq)
            frame_t = pd.read_csv(ft_path, sep="\t")["t"].to_numpy()
            sampling_period = np.diff(frame_t)
            self.sf = 1 / sampling_period.mean()
        else:
            end = self.n_img * (1 / self.sf)
            frame_t = np.linspace(0, end, endpoint=False, num=self.n_img)

        self.logger.info(f"Estimated sampling frequency of video: {self.sf:.2f})")
        self.frame_t = frame_t

    def _rescale(self, img):
        """Rescales an image with a scaling factor `scaling`."""
        img = rescale(
            img, self.scaling, preserve_range=True, anti_aliasing=True, channel_axis=2
        )
        img = img.round().astype(np.uint8)
        return img

    def loop(self, return_index=True):
        """Loops across frames of a video.

        Parameters
        ----------
        return_index : bool
            Whether to return the frame index and the image; if ``False``,
            only the image is returned

        Yields
        ------
        img : np.ndarray
            Numpy array (dtype: ``np.uint8``) of shape width x height x 3 (RGB)
        idx : int
            Optionally (when ``return_index`` is set to ``True``), returns the index of
            the currently looped frame
        """
        reader = imageio.get_reader(self.path, mode='I')
        desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")
        self.stop_loop_ = False

        if self.logger.level <= logging.INFO:
            iter_ = tqdm(reader, desc=f"{desc} Recon frames", total=self.n_img)
        else:
            iter_ = reader

        i = 0
        for img in iter_:
            
            if img.ndim == 2:
                # If we have grayscale data, convert to RGB
                if i == 0:
                    # Only log the first time to avoid clutter
                    self.logger.warn("Data seems grayscale; converting to RGB")
                    img = cv2.cvtColor(img ,cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:
                # If we have an RGBA image, just trim off 4th dim
                img = img[..., :3]
            else:
                if img.ndim == 3 and img.shape[2] != 3:
                    self.logger.error(f"Data has the wrong shape {img.shape}, probably" 
                                       "going to crash!")
           
            if self.stop_loop_:
                break

            if self.scaling is not None:
                img = self._rescale(img)
            
            i += 1
            if return_index:
                idx = i - 1
                yield idx, img
            else:
                yield img

        reader.close()

        if hasattr(self, "writer"):
            self.writer.close()

    def stop_loop(self):
        """Stops the loop over frames (in self.loop)."""
        self.stop_loop_ = True

    def create_writer(self, path, idf="crop", ext="gif"):
        """Creates a imageio writer object, which can for example
        be used to save crop parameters on top of each frame of
        a video."""
        self.writer = imageio.get_writer(path + f"_{idf}.{ext}", mode="I", fps=self.sf)

    def write(self, img):
        """Adds image to writer."""
        self.writer.append_data(img)

    def get_metadata(self):
        """Returns all (meta)data needed for initialization
        of a Data object."""

        return {
            "frame_t": self.frame_t,
            "img_size": self.img_size,
            "events": self.events,
            "sf": self.sf,
            "loglevel": self.logger.level
        }


def load_h5(path):
    """Convenience function to load a hdf5 file and immediately initialize the correct
    data class.

    Parameters
    ----------
    path : str
        Path to an HDF5 file

    Returns
    -------
    data : ``data.BaseData`` subclass
        An object with a class derived from ``data.BaseData``
        (like ``MediapipeData``, or ``FlameData``)
        
    Examples
    --------
    Load in HDF5 data reconstructed by Mediapipe:
    
    >>> from medusa.data import get_example_h5
    >>> path = get_example_h5(load=False)
    >>> data = load_h5(path)    
    """

    from .core import MODEL2CLS

    # peek at recon model
    with h5py.File(path, "r") as f_in:
        rmn = f_in.attrs["recon_model"]

    data = MODEL2CLS[rmn].load(path)
    return data

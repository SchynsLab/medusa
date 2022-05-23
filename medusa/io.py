""" Module with functionality (mostly) for working with video data. 
The `VideoData` class allows for easy looping over frames of a video file,
which is used in the reconstruction process (e.g., in the ``videorecon`` function).
"""

import cv2
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
        self.events = events
        self.scaling = scaling
        self.logger = get_logger()
        self.logger.setLevel(loglevel)
        self._validate()
        self._extract_metadata()

        if find_files:
            self._find_events()
            self._find_frame_t()

    def _validate(self):
        if not self.path.is_file():
            raise ValueError(f"File {self.path} does not exist!")

        sfx = self.path.suffix
        if sfx != ".mp4":
            raise ValueError(f"Only mp4 videos are supported, not {sfx[1:]}!")

    def _extract_metadata(self):
        """Extracts some metadata from the video needed for preprocessing
        functionality later on."""

        reader = imageio.get_reader(self.path)
        self.sf = reader.get_meta_data()["fps"]  # sf = sampling frequency
        self.img_size = reader.get_meta_data()["size"]

        # Use cv2 to get n_frames (not possible with imageio ...)
        cap = cv2.VideoCapture(str(self.path))
        self.n_img = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()
        reader.close()

    def _find_events(self):
        """If not already supplied upon initialization of the Video object,
        this method trieds to find an event file associated with the video,
        which is assumed to have the same name as the video but ending in
        "_events.tsv" instead of ".mp4". If it's not found, it is simply
        ignored."""

        if self.events is None:
            # check for associated TSV path
            self.events = Path(str(self.path).replace(".mp4", "_events.tsv"))

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
        ft_path = Path(str(self.path).replace(".mp4", "_frametimes.tsv"))
        if ft_path.is_file():
            # We're going to use the average number of frames
            # per second as FPS (and equivalently `sf`, sampling freq)
            frame_t = pd.read_csv(ft_path, sep="\t")["t"].to_numpy()
            sampling_period = np.diff(frame_t)
            self.sf = 1 / sampling_period.mean()
        else:
            end = self.n_img * (1 / self.sf)
            frame_t = np.linspace(0, end, endpoint=False, num=self.n_img)

        self.logger.info(
            f"Estimated sampling frequency of video: {self.sf:.2f})"
        )

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
            Whether to return the frame index and the image; if `False`,
            only the image is returned

        Yields
        ------
        img : np.ndarray
            Numpy array (dtype: `np.uint8`) of shape width x height x 3 (RGB)
        """
        reader = imageio.get_reader(self.path)
        desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")
        self.stop_loop_ = False

        if self.logger.level <= logging.INFO:
            iter_ = tqdm(reader, desc=f"{desc} Recon frames", total=self.n_img)
        else:
            iter_ = reader

        i = 0
        for img in iter_:

            if self.stop_loop_:
                break

            if self.scaling is not None:
                img = self._rescale(img)

            i += 1
            if return_index:
                yield i - 1, img
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


try:
    from mne.epochs import EpochsArray as EpochsArrayBase
except ImportError:
    EpochsArrayBase = object
        

class EpochsArray(EpochsArrayBase):
    """Custom EpochsArray, with some extra functionality to interact with
    medusa.

    Parameters
    ----------
    args : list
        Positional parameters to be passed to initialization of the
        MNE EPochsArray (the base class)
    kwargs : list
        Keyword parameters to be passed to initialization of the
        MNE EPochsArray (the base class)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_medusa(cls, v, sf, events=None, frame_t=None, tmin=-0.5,
                    includes_motion=False):
        """Classmethod to initalize an EpochsArray from medusa data.

        Parameters
        ----------
        v : np.ndarray
            A 4D numpy array of shape N (events/trails) x T (time points)
            x nV (number of vertices) x 3 (X, Y, Z)
        sf : float
            Sampling frequency of the data (`v`)
        events : pd.DataFrame
            events : pd.DataFrame
            A BIDS-style DataFrame with event (trial) information,
            with at least the columns 'onset' and 'trial_type'
        frame_t : np.ndarray
            A 1D numpy array with the onset of each frame from
            the video that was reconstructed
        tmin : float
            Start (in seconds) of each epoch relative to stimulus onset
        includes_motion : bool
            Whether the data (`v`) also includes the epoched motion parameters;
            if so, it is assumed that the last 12 values in the third dimension
            of `v` represents the motion parameters

        Returns
        -------
        An instance of the EpochsArray class
        """
        
        try:
            import mne
        except ImportError:
            raise ValueError("MNE is not installed!")    
            
        v = v.copy()

        # N (trails), T (time points), nV (number of vertices)
        N, T, nV = v.shape[:3]

        # Flatten vertices and coord (XYZ) dimensions
        v = v.reshape((N, T, -1))

        # N x T x (V x 3) --> N x (V x 3) x T
        # (as is expected by MNE)
        v = np.transpose(v, (0, 2, 1))

        if includes_motion:
            ch_names = [
                f"v{i}_{c}" for i in range((nV - 12) // 3) for c in ["x", "y", "z"]
            ]
            ch_names += [
                "xt",
                "yt",
                "zt",
                "xr",
                "yr",
                "zr",
                "xs",
                "ys",
                "zs",
                "xsh",
                "ysh",
                "zsh",
            ]
        else:
            ch_names = [f"v{i}_{c}" for i in range(nV // 3) for c in ["x", "y", "z"]]

        info = mne.create_info(
            # vertex 0 (x), vertex 0 (y), vertex 0 (z), vertex 1 (x), etc
            ch_names=ch_names,
            ch_types=["misc"] * v.shape[1],
            sfreq=sf,
        )

        if events is not None:
            events_, event_id = cls.events_to_mne(events, frame_t)
        else:
            events_, event_id = None, None

        return cls(
            v, info, tmin=tmin, events=events_, event_id=event_id, verbose="WARNING"
        )

    @staticmethod
    def events_to_mne(events, frame_t):
        """Converts events DataFrame to (N x 3) array that
        MNE expects.

        Parameters
        ----------
        events : pd.DataFrame
            A BIDS-style DataFrame with event (trial) information,
            with at least the columns 'onset' and 'trial_type'
        frame_t : np.ndarray
            A 1D numpy array with the onset of each frame from
            the video that was reconstructed; necessary for
            converting event onsets in seconds to event onsets
            in samples (TODO: use sf for this?)

        Returns
        -------
        events_ : np.ndarray
            An N (number of trials) x 3 array, with the first column
            indicating the sample *number* (not time) and the third
            column indicating the sample condition (see the returned
            `event_id` dictionary for the mapping between condition number
            and string representation)
        event_id : dict
            A dictionary with condition strings as keys and condition numbers
            as values; the values correspond to the third column of `events_`
        """

        event_id = {k: i for i, k in enumerate(events["trial_type"].unique())}
        events_ = np.zeros((events.shape[0], 3))
        for i, (_, ev) in enumerate(events.iterrows()):
            events_[i, 2] = event_id[ev["trial_type"]]
            t_diff = np.abs(frame_t - ev["onset"])
            events_[i, 0] = np.argmin(t_diff)

            if np.min(t_diff) > 0.05:
                min_ = np.min(t_diff).round(4)
                raise ValueError(
                    f"Nearest sample is {min_} seconds away "
                    f"for trial {i+1}; try resampling the data to a "
                    "higher resolution!"
                )

        events_ = events_.astype(np.int64)

        return events_, event_id

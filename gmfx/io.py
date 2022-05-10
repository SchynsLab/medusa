import cv2
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from skimage.transform import rescale

from .utils import get_logger

logger = get_logger()


class VideoData:
    """" Contains (meta)data and functionality associated
    with video files (mp4 files only currently). 
    
    Parameters
    ----------
    path : str, Path
        Path to mp4 file
    events : str, Path
        Path to a TSV file with event information (optional);
        should contain at least the columns 'onset' and 'trial_type'

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
    def __init__(self, path, events=None):
        self.path = Path(path)
        self.events = events
        self._validate()
        self._extract_metadata()
        self._find_events()
        self._find_frame_t()

    def _validate(self):
        if not self.path.is_file():
            raise ValueError(f"File {self.path} does not exist!")

        sfx = self.path.suffix
        if sfx != '.mp4':
            raise ValueError(f"Only mp4 videos are supported, not {sfx[1:]}!")

    def _extract_metadata(self):
        """ Extracts some metadata from the video needed for preprocessing
        functionality later on. """
 
        reader = imageio.get_reader(self.path)
        self.sf = reader.get_meta_data()['fps']  # sf = sampling frequency
        self.img_size = reader.get_meta_data()['size']
        
        # Use cv2 to get n_frames (not possible with imageio ...)
        cap = cv2.VideoCapture(str(self.path))
        self.n_img = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        reader.close()

    def _find_events(self):
        """ If not already supplied upon initialization of the Video object,
        this method trieds to find an event file associated with the video,
        which is assumed to have the same name as the video but ending in
        "_events.tsv" instead of ".mp4". If it's not found, it is simply
        ignored. """

        if self.events is None:
            # check for associated TSV path
            self.events = Path(str(self.path).replace('.mp4', '_events.tsv'))
    
        if self.events.is_file():
            self.events = pd.read_csv(self.events, sep='\t')
        else:
            self.events = None
            logger.warning(f"Did not find events file for video {self.path}!")

    def _find_frame_t(self):
        """ Looks for a "frame times" file associated with the video, 
        which is assumed to have the same name as the video but ending
        in "_frametimes.tsv" instead of ".mp4". If it's not found,
        the frame times are assumed to be constant and a multiple of
        the fps (frames per second). """
        
        # Check if frame times (ft) and events (ev) files exist    
        ft_path = Path(str(self.path).replace('.mp4', '_frametimes.tsv'))
        if ft_path.is_file():
            # We're going to use the average number of frames
            # per second as FPS (and equivalently `sf`, sampling freq)
            frame_t = pd.read_csv(ft_path, sep='\t')['t'].to_numpy()
            sampling_period = np.diff(frame_t)
            self.sf, sf_std = 1 / sampling_period.mean(), 1 / sampling_period.std()
            logger.info(f"Average FPS/sampling frequency: {self.sf:.2f} (SD: {sf_std:.2f})")
        else:
            logger.warning(f"Did not find frame times file for {self.path} "
                           f"assuming constant FPS/sampling frequency ({self.sf})!")
        
            end = self.n_img * (1 / self.sf)
            frame_t = np.linspace(0, end, endpoint=False, num=self.n_img)

        self.frame_t = frame_t
    
    def loop(self, scaling=None, return_index=True):
        """ Loops across frames of a video.
        
        Parameters
        ----------
        scaling : float
            If not `None` (default), rescale image with this factor
            (e.g., 0.25 means reduce image to 25% or original)
        return_index : bool
            Whether to return the frame index and the image; if `False`,
            only the image is returned
            
        Yields
        ------
        img : np.ndarray
            Numpy array (dtype: `np.uint8`) of shape width x height x 3 (RGB)
        """
        reader = imageio.get_reader(self.path)
        desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ] ')
        self.stop_loop_ = False

        i = 0
        for img in tqdm(reader, desc=f"{desc} Recon frames", total=self.n_img):

            if self.stop_loop_:
                break
            
            if scaling is not None:
                img = rescale(img, scaling, preserve_range=True, anti_aliasing=True,
                              channel_axis=2).round().astype(np.uint8)
    
            i += 1
            if return_index:                
                yield i - 1, img
            else:
                yield img

        reader.close()
        
        if hasattr(self, 'writer'):
            self.writer.close()

    def stop_loop(self):
        """ Stops the loop over frames (in self.loop). """
        self.stop_loop_ = True

    def create_writer(self, path, idf='crop', ext='gif'):
        """ Creates a imageio writer object, which can for example
        be used to save crop parameters on top of each frame of
        a video. """ 
        self.writer = imageio.get_writer(path + f'_{idf}.{ext}', mode='I', fps=self.sf)
        
    def write(self, img):
        """ Adds image to writer. """
        self.writer.append_data(img)
        
    def get_metadata(self):
        """ Returns all (meta)data needed for initialization
        of a Data object. """
        
        return {
            'frame_t': self.frame_t,
            'img_size': self.img_size,
            'events': self.events,
            'sf': self.sf
        }

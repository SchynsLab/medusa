import os
import torch
import h5py
import imageio
import numpy as np
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.decomposition import PCA

from .constants import FACES
from .recon.emoca.utils import tensor2image
from .render.renderer import SRenderY
from .utils import get_logger

logger = get_logger()


class Data:
    """ Generic Data class to store, load, and save vertex/face data. 
    
    Parameters
    ----------
    v : ndarray
        Numpy array of shape T (time points) x nV (no. vertices) x 3 (x/y/z)
    motion : ndarray
        Numpy array of shape T (time points) x 6 (global rot x/y/z, trans x/y, scale z)
    f : ndarray
        Integer numpy array of shape nF (no. faces) x 3 (vertices per face)
    frame_t : ndarray
        Numpy array of length T (time points) with "frame times", i.e.,
        onset of each frame (in seconds) from the video
    events : pd.DataFrame
        A Pandas DataFrame with `N` rows corresponding to `N` separate trials
        and at least two columns: 'onset' and 'trial_type' (optional: 'duration')
    sf : int, float
        Sampling frequency of video
    dense : bool
        Whether we're using FLAME's dense (True) or coarse (False) mesh
    motion_cols : list[str]
        List with names of motion parameters
        
    Attributes
    ----------
    path : str
        Path to which the data is (eventually) saved
    """
    def __init__(self, v, motion=None, f=None, frame_t=None, events=None,
                 sf=None, dense=False, motion_cols=None):

        self.v = v
        self.motion = motion
        self.f = f
        self.frame_t = frame_t
        self.events = events
        self.sf = sf
        self.dense = dense
        self.motion_cols = motion_cols
        self.path = None
        self._check()

    def _check(self):
        """ Does some checks to make sure the data works with
        the renderer and other stuff. """
        # Renderer expects torch floats (float32), not double (float64)
        self.v = self.v.astype(np.float32)

        if self.f is None:
            self.f = FACES['dense'] if self.dense else FACES['coarse']
            self.f = self.f.astype(np.int32)

        if self.frame_t is not None:
            if self.frame_t.size != self.v.shape[0]:
                raise ValueError("Number of frame times does not equal "
                                 "number of vertex time points!")
                
        if self.motion is not None:
            if self.frame_t.size != self.motion.shape[0]:
                raise ValueError("Number of frame times does not equal " 
                                 "number of motion time points!")

            if self.motion_cols is None:
                if self.motion.shape[1] == 6:
                    self.motion_cols = ['rot_x', 'rot_y', 'rot_z', 'trans_x',
                                        'trans_y', 'trans_z']

    @classmethod
    def load(cls, path):
        """ Loads a hdf5 file from disk and returns a Data object. """
        
        with h5py.File(path, "r") as f_in:
            v = f_in['v'][:]

            motion_cols = None            
            if 'motion' in f_in:
                motion = f_in['motion'][:]
                if 'motion_cols' in f_in['motion'].attrs:
                    motion_cols = f_in['motion'].attrs['motion_cols']
            else:
                motion = None

            f = f_in['f'][:]            
            frame_t = f_in['frame_t'][:]
            dense = f_in['v'].attrs['dense']
            sf = f_in['frame_t'].attrs['sf']
            load_events = 'events' in f_in

        if load_events:
            events = pd.read_hdf(path, key='events')
        else:
            events = None
        
        data = cls(v, motion, f, frame_t, events, sf, dense, motion_cols)
        data.path = str(path)
        return data
    
    def events_to_mne(self):
        """ Converts events DataFrame to (N x 3) array that
        MNE expects. 
        
        Returns
        -------
        events : np.ndarray
            An N (number of trials) x 3 array, with the first column
            indicating the sample *number* indicating the 
        """
        
        if self.events is None:
            raise ValueError("There are no events associated with this data object!")
            
        event_id = {k: i for i, k in enumerate(self.events['trial_type'].unique())}
        events = np.zeros((self.events.shape[0], 3))
        for i, (_, ev) in enumerate(self.events.iterrows()):
            events[i, 2] = event_id[ev['trial_type']]
            events[i, 0] = np.argmin(np.abs(self.frame_t - ev['onset']))
            
            if events[i, 0] > 0.05:
                raise ValueError(f"Nearest sample > 0.05 seconds away for trial {i+1}; "
                                  "Try resampling the data to a higher resolution!")
        
        return events, event_id
       
    def to_mne_rawarray(self):
        """ Creates an MNE `RawArray` object from the vertices (`v`)."""
        try:
            import mne
        except ImportError:
            raise ValueError("The mne package is not installed!")

        T, nV = self.v.shape[:2]
        info = mne.create_info(
            # vertex 0 (x), vertex 0 (y), vertex 0 (z), vertex 1 (x), etc
            ch_names=[f'v{i}_{c}' for i in range(nV) for c in ['x', 'y', 'z']],
            ch_types=['misc'] * np.prod(self.v.shape[1:]),
            sfreq=self.sf
        )
        return mne.io.RawArray(self.v.reshape(T, -1), info)
     
    def save(self, path):
        """ Saves data to disk as a hdf5 file. """

        out_dir = op.dirname(path)
        if not op.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with h5py.File(path, 'w') as f_out:
            for attr in ['v', 'motion', 'f', 'frame_t']:
                data = getattr(self, attr)
                if data is not None:
                    f_out.create_dataset(attr, data=data)

            if self.motion_cols is not None:
                # We can safely assume that f_out['motion'] exists
                f_out['motion'].attrs['motion_cols'] = self.motion_cols

            f_out['v'].attrs['dense'] = self.dense
            f_out['frame_t'].attrs['sf'] = self.sf

        # Note to self: need to do this outside h5py.File context,
        # because Pandas assumes a buffer or path
        if self.events is not None:
            self.events.to_hdf(path, key='events', mode='a')

    def render_video(self, f_out, renderer=None, device='cuda'):
        """ Renders a video of each time point of `self.v`.
        
        Parameters
        ----------
        renderer : SRenderY
            An instance of `gmfx.render.renderer.SRenderY`; If `None`,
            it is instantiated here
        device : str
            Either 'cuda' (GPU) or 'cpu'; only used when `renderer` is `None`,
            otherwise ignored
        """
        if renderer is None:
            templ_obj = Path(__file__).parent / 'data/head_template.obj'
            renderer = SRenderY(224, templ_obj, device=device)

        # Also create video of "world space" reconstruction
        writer = imageio.get_writer(f_out, mode='I', fps=self.sf)
        desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ] ')
        for i in tqdm(range(self.v.shape[0]), desc=f'{desc} Render shape'):
            V = torch.from_numpy(self.v[i, ...]).to(renderer.device)
            V = torch.unsqueeze(V, 0)
            V_trans = V.clone() * 7  # zoom, should be param
            V_trans[..., 1:] = -V_trans[..., 1:]
            res = renderer.render_shape(V, V_trans, h=224, w=224, images=None)
            writer.append_data(tensor2image(res[0], bgr=False))

        writer.close()        

    def plot_data(self, f_out, plot_motion=True, plot_pca=True, n_pca=3):        
        """ Creates a plot of the motion (rotation & translation) parameters
        over time and, optionally, the first `n_pca` PCA components of the 
        reconstructed vertices. For FLAME estimates, these parameters are
        relative to the canonical model, so the estimates are plotted relative
        to the value of the first frame.

        Parameters
        ----------
        f_out : str, Path
            Where to save the plot to (a png file)
        plot_motion : bool
            Whether to plot the motion parameters
        plot_pca : bool
            Whether to plot the `n_pca` PCA-transformed traces of the data (`self.v`)
        n_pca : int
            How many PCA components to plot
        """
    
        if self.motion is None:
            logger.warn("No motion params available; setting plot_motion to False")
            plot_motion = False
    
        if not plot_motion and not plot_pca:
            raise ValueError("`plot_motion` and `plot_pca` cannot both be False")
    
        if plot_motion and plot_pca:
            nrows = self.motion.shape[1] + 1
        elif plot_motion:
            nrows = self.motion.shape[1]
        else:
            nrows = 1
            
        fig, axes = plt.subplots(nrows=nrows, sharex=True, figsize=(10, 2 * nrows))
        if nrows == 1:
            axes = [axes]

        t = self.frame_t  # time in sec.
        
        if plot_motion:
            for i, name in enumerate(self.motion_cols):
                axes[i].plot(t, self.motion[:, i] - self.motion[0, i])
                axes[i].set_ylabel(name, fontsize=10)
        
        if plot_pca:
            # Perform PCA on flattened vertex array (T x (nV x 3))
            pca = PCA(n_components=n_pca)
            v_pca = pca.fit_transform(self.v.reshape(self.v.shape[0], -1))
            axes[-1].plot(t, v_pca)
            axes[-1].legend([f'PC{i+1}' for i in range(n_pca)])
            axes[-1].set_ylabel('Movement (A.U.)', fontsize=10)
        
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.grid(True)

            if self.events is not None:
                # Also plot a dashed vertical line for each trial (with a separate
                # color for each condition)
                for i, tt in enumerate(self.events['trial_type'].unique()):
                    max_t = t.max()
                    ev = self.events.query("trial_type == @tt & onset < @max_t")
                    for _, ev_i in ev.iterrows():
                        ax.axvline(ev_i['onset'], ls='--', c=plt.cm.viridis(i))

        axes[-1].set_xlabel('Time (frames)', fontsize=10)
        fig.savefig(f_out)
        plt.close()

    def __len__(self):
        return self.v.shape[0]
    
    def __getitem__(self, idx):
        return self.v[idx, :, :]
    
    def __setitem__(self, idx, v):
        self.v[idx, ...] = v
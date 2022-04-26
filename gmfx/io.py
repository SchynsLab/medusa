import os
os.environ['DISPLAY'] = ':0.0'

import cv2
import h5py
import imageio
import numpy as np
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from trimesh import Trimesh
from datetime import datetime
from skimage.transform import warp
from sklearn.decomposition import PCA
from pyrender import Scene, Mesh, OffscreenRenderer
from pyrender import SpotLight, OrthographicCamera, Node

from .constants import FACES
from .utils import get_logger

logger = get_logger()


class BaseData:
    """ Base Data class with attributes and methods common to all Data
    classes (such as FlameData, MediapipeData, etc.).
    
    Warning: objects should never be initialized with this class directly,
    only when calling super().__init__() from the subclass (like `FlameData`).

    Parameters
    ----------
    v : ndarray
        Numpy array of shape T (time points) x nV (no. vertices) x 3 (x/y/z)
    motion : ndarray
        Numpy array of shape T (time points) x 6 (global rot x/y/z, trans x/y, scale z)
    f : ndarray
        Integer numpy array of shape nF (no. faces) x 3 (vertices per face); can be
        `None` if working with landmarks/vertices only
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
    recon_model_name : str
        Name of reconstruction model
    path : str
        Path where the data is saved; if initializing a new object (rather than
        loading one from disk), this should be `None`
    """
    def __init__(self, v, motion=None, f=None, frame_t=None, events=None,
                 sf=None, recon_model_name=None, path=None, **kwargs):

        self.v = v
        self.motion = motion
        self.f = f
        self.frame_t = frame_t
        self.events = events
        self.sf = sf
        self.recon_model_name = recon_model_name
        self.path = path
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._check()
    
    def _check(self):
        """ Does some checks to make sure the data works with
        the renderer and other stuff. """

        # Renderer expects torch floats (float32), not double (float64)
        self.v = self.v.astype(np.float32)

        if self.frame_t is not None:
            if self.frame_t.size != self.v.shape[0]:
                raise ValueError("Number of frame times does not equal "
                                 "number of vertex time points!")
                
        if self.motion is not None:
            if self.frame_t.size != self.motion.shape[0]:
                raise ValueError("Number of frame times does not equal " 
                                 "number of motion time points!")
    
    @staticmethod
    def load(path):
        """ Loads a hdf5 file from disk and returns a Data object. """
        
        with h5py.File(path, "r") as f_in:
            v = f_in['v'][:]

            if 'motion' in f_in:
                motion = f_in['motion'][:]
            else:
                motion = None

            if 'f' in f_in:
                f = f_in['f'][:]            
            else:
                f = None

            frame_t = f_in['frame_t'][:]
            recon_model_name = f_in.attrs['recon_model_name']
            sf = f_in['frame_t'].attrs['sf']
            load_events = 'events' in f_in

        if load_events:
            events = pd.read_hdf(path, key='events')
        else:
            events = None

        return {'v': v, 'motion': motion, 'f': f, 'frame_t': frame_t,
                'events': events, 'sf': sf,
                'recon_model_name': recon_model_name, 'path': path}
    
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
        import mne
        T, nV = self.v.shape[:2]
        info = mne.create_info(
            # vertex 0 (x), vertex 0 (y), vertex 0 (z), vertex 1 (x), etc
            ch_names=[f'v{i}_{c}' for i in range(nV) for c in ['x', 'y', 'z']],
            ch_types=['misc'] * np.prod(self.v.shape[1:]),
            sfreq=self.sf
        )
        return mne.io.RawArray(self.v.reshape(T, -1), info)
     
    def to_mne_epochs(self):
        import mne

        # TODO: create epochs and init mne.Epochs        
     
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

            f_out.attrs['data_class'] = self.__class__.__name__
            f_out.attrs['recon_model_name'] = self.recon_model_name
            f_out['frame_t'].attrs['sf'] = self.sf

        # Note to self: need to do this outside h5py.File context,
        # because Pandas assumes a buffer or path
        if self.events is not None:
            self.events.to_hdf(path, key='events', mode='a')

    def render_video(self, *args, **kwargs):
        """ Should be implemented in subclass! """
        raise NotImplementedError

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
            motion_cols = ['scale', 'x_trans', 'y_trans', 'rot x', 'rot y', 'rot z']
            for i, name in enumerate(motion_cols):
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
                        ax.axvline(ev_i['onset'], ls='--', c=plt.cm.tab20(i))

        axes[-1].set_xlabel('Time (frames)', fontsize=10)
        fig.savefig(f_out)
        plt.close()

    def __len__(self):
        return self.v.shape[0]
    
    def __getitem__(self, idx):
        return self.v[idx, :, :]
    
    def __setitem__(self, idx, v):
        self.v[idx, ...] = v
        

class FlameData(BaseData):
    
    def __init__(self, *args, **kwargs):

        kwargs['f'] = FACES['coarse']
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path):
        
        init_kwargs = super().load(path)
        if init_kwargs['f'] is None:
            init_kwargs['f'] = FACES['coarse']

        return cls(**init_kwargs)

    def render_video(self, f_out, viewport=(224, 224), video=None, zoom_out=4):

        if video is not None:
            # Plot face on top of video, so need to load in video
            reader = imageio.get_reader(video)
            out_size = reader.get_meta_data()['source_size']
            out_size = (out_size[1], out_size[0])

        scene = Scene(bg_color=[0, 0, 0])
        camera = OrthographicCamera(xmag=1, ymag=1)
        scene.add_node(Node(camera=camera, translation=(0, 0, zoom_out)))
        light = SpotLight(intensity=50)
        scene.add_node(Node(light=light, translation=(0, 0, zoom_out)))
        renderer = OffscreenRenderer(viewport_width=viewport[0], viewport_height=viewport[1])
        
        writer = imageio.get_writer(f_out, mode='I', fps=self.sf)
        desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ] ')
        
        for i in tqdm(range(len(self)), desc=f'{desc} Render shape'):

            mesh = Mesh.from_trimesh(Trimesh(self.v[i, :, :], self.f))
            mesh_node = Node(mesh=mesh)
            scene.add_node(mesh_node)
            img, _ = renderer.render(scene)
            img = img.copy()  # otherwise it's read only

            if video is not None:
                background = reader.get_data(i)
                img = warp(img, self.tform[i, :, :], output_shape=out_size, preserve_range=True)
                img = img.astype(np.uint8)
                img[img == 0] = background[img == 0]

            writer.append_data(img)
            scene.remove_node(mesh_node)

        renderer.delete()
        writer.close()
        
        if video is not None:
            reader.close()   


class MediapipeData(BaseData):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FANData(BaseData):

    @classmethod
    def load(cls, path):
        
        init_kwargs = super().load(path)
        return cls(**init_kwargs)

        
    def render_video(self, f_out, video=None, margin=25):
        
        if video is not None:
            # Plot face on top of video, so need to load in video
            reader = imageio.get_reader(video)
            v = self.v
        else:
            v = self.v
            h = int(v[:, :, 0].max()) - int(v[:, :, 0].min()) + margin * 2
            w = int(v[:, :, 1].max()) - int(v[:, :, 1].min()) + margin * 2
            v = v - v.min(axis=(0, 1)) + margin

        writer = imageio.get_writer(f_out, mode='I', fps=self.sf)
        desc = datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ] ')

        for i in tqdm(range(len(self)), desc=f'{desc} Render shape'):
            
            # fig, ax = plt.subplots()
            if video is not None:
                background = reader.get_data(i)
            else:
                background = np.zeros((w, h, 3)).astype(np.uint8)

            for ii in range(self.v.shape[1]):
                cv2.circle(background, np.round(v[i, ii, :2]).astype(int), radius=2, color=(255, 0, 0), thickness=3)
            
            writer.append_data(background)
        
        writer.close()
        if video is not None:
            reader.close()            


MODEL2CLS = {
    'emoca': FlameData,
    'deca': FlameData,
    'mediapipe': MediapipeData,
    'FAN-3D': FANData
}


def load_h5(path):
    
    # peek at recon model
    with h5py.File(path, "r") as f_in:
        rmn = f_in.attrs['recon_model_name']
        
    return MODEL2CLS[rmn].load(path)
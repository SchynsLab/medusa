""" Module with core 4D functionality of the ``medusa`` package, most importantly the
``*4D`` classes. The ``Base4D`` class defines a template class from which
model-specific classes (such as ``Flame4D``) inherit. Objects initialized from these
classes store reconstructed data from videos and other (meta)data needed to further
process, analyze, and visualize it.

The reconstructed data from each model supported by ``medusa`` is stored in an object from
a specific class which inherits from ``Base4D``. For example, reconstructed data from
`mediapipe <https://google.github.io/mediapipe/solutions/face_mesh.html>`_ is stored
in using the ``Mediapipe4D`` class. Other classes include the ``Fan4D`` for
reconstructions from `FAN <https://github.com/1adrianb/face-alignment>`_ and
``Flame4D`` for reconstructions from models using the `FLAME topology <https://flame.is.tue.mpg.de/>`_
(such as `EMOCA <https://emoca.is.tue.mpg.de/>`_).

The data can be saved to disk as a `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
file (using `h5py <http://www.h5py.org/>`_) with the ``save`` method and loaded from
disk using the ``load`` (static)method.   
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import h5py
import logging
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from skimage.transform import rescale
from sklearn.decomposition import PCA
from trimesh.transformations import decompose_matrix, compose_matrix

from .utils import get_logger
from .render import Renderer


class Base4D:
    """Base Data class with attributes and methods common to all 4D data classes
    (such as ``Flame4D``, ``Mediapipe4D``, etc.).

    Warning: objects should never be initialized with this class directly,
    only when calling ``super().__init__()`` from the subclass (like ``Flame4D``). Note,
    though, that the initialization parameters are the same for every class that
    inherits from ``Base4D``.

    Parameters
    ----------
    v : ndarray
        Numpy array of shape T (time points) x nV (no. vertices) x 3 (x/y/z)
    f : ndarray
        Integer numpy array of shape nF (no. faces) x 3 (vertices per face); can be
        `None` if working with landmarks/vertices only
    mat : ndarray
        Numpy array of shape T (time points) x 4 x 4 (affine matrix) representing
        the 'world' (or 'model') matrix for each time point
    cam_mat : ndarray
        Numpy array of shape 4x4 (affine matrix) representing the camera matrix
    frame_t : ndarray
        Numpy array of length T (time points) with "frame times", i.e.,
        onset of each frame (in seconds) from the video
    events : pd.DataFrame
        A Pandas DataFrame with `N` rows corresponding to `N` separate trials
        and at least two columns: 'onset' and 'trial_type' (optional: 'duration')
    sf : int, float
        Sampling frequency of video
    recon_model_name : str
        Name of reconstruction model
    space : str
        The space the vertices are currently in; can be either 'local' or 'world'
    path : str
        Path where the data is saved; if initializing a new object (rather than
        loading one from disk), this should be `None`
    loglevel : int
        Logging level of current logger
    """

    def __init__(
        self,
        v,
        f,
        mat=None,
        cam_mat=None,
        frame_t=None,
        events=None,
        sf=None,
        img_size=(512, 512),
        recon_model_name=None,
        space="world",
        path=None,
        loglevel=20
    ):

        self.v = v
        self.f = f
        self.mat = mat
        self.cam_mat = cam_mat
        self.frame_t = frame_t
        self.events = events
        self.sf = sf
        self.img_size = img_size
        self.recon_model_name = recon_model_name
        self.space = space
        self.path = path
        self.logger = get_logger()
        self.logger.setLevel(int(loglevel))
        self._check()

    def _check(self):
        """Does some checks to make sure the data works with
        the renderer and other stuff."""

        # Renderer expects torch floats (float32), not double (float64)
        self.v = self.v.astype(np.float32)
        T = self.v.shape[0]
        if self.frame_t.size > T:
            self.logger.warn(
                f"More frame times {self.frame_t.size} than vertex time points ({T}); trimming ..."
            )
            self.frame_t = self.frame_t[:T]

        if self.mat is not None:
            if self.mat.shape[0] != T:
                mT = self.mat.shape[0]
                self.logger.warn(
                    f"More mats ({mT}) than vertex time points ({T}); trimming ..."
                )
                self.mat = self.mat[:T, :, :]

        if self.events is not None:
            # Trim off any events recorded after video recording ended
            # (otherwise time series plots look weird)
            max_t = self.frame_t[-1]
            self.events = self.events.query("onset < @max_t")

        if self.space not in ["local", "world"]:
            raise ValueError("`space` should be either 'local' or 'world'!")

    def project_to_68_landmarks(self):
        """ Projects to 68 landmark set. """
        
        if self.recon_model_name not in ['mediapipe', 'deca-coarse', 'emoca-coarse']:
            raise ValueError("Can only project to 68 landmarks for mediapipe and "
                             "(coarse) FLAME-based topologies!")

        if self.recon_model_name == 'mediapipe':
            fname = 'mediapipe_lmk68_embedding.npz'
        else:
            fname = 'flame_lmk68_embedding.npz'

        emb = np.load(Path(__file__).parents[1] / f'data/{fname}')
        vf = self.v[:, self.f[emb['lmk_faces_idx']]]
        v_proj = np.sum(vf * emb['lmk_bary_coords'][:, :, None], axis=1)
        return v_proj

    def mats2params(self, to_df=True):
        """Transforms a time series (of length T) 4x4 affine matrices to a
        pandas DataFrame with a time series of T x 12 affine parameters
        (translation XYZ, rotation XYZ, scale XYZ, shear XYZ).
        
        Parameters
        ----------
        to_df : bool
            Whether to return the parameters as a pandas ``DataFrame`` or
            not (in which case it's returned as a numpy array)

        Returns
        -------
        params : pd.DataFrame, np.ndarray
            Either a ``DataFrame`` or numpy array, depending on the ``to_df`` parameter
            
        Examples
        --------
        Convert the sequences of affine matrices to a 2D numpy array:
        
        >>> from medusa.data import get_example_h5
        >>> data = get_example_h5(load=True, model="mediapipe")
        >>> params = data.mats2params(to_df=False)
        >>> params.shape
        (232, 12)
        """

        if self.mat is None:
            raise ValueError(
                "Cannot convert matrices to parameters because "
                "there are no matrices (self.mat is None)!"
            )

        T = self.mat.shape[0]
        params = np.zeros((T, 12))
        for i in range(T):
            scale, shear, angles, trans, _ = decompose_matrix(self.mat[i, :, :])
            params[i, :3] = trans
            params[i, 3:6] = np.rad2deg(angles)
            params[i, 6:9] = scale
            params[i, 9:12] = shear

        if to_df:
            cols = [
                "Trans. X",
                "Trans. Y",
                "Trans Z.",
                "Rot. X (deg)",
                "Rot. Y (deg)",
                "Rot. Z (deg)",
                "Scale X (A.U.)",
                "Scale Y (A.U.)",
                "Scale Z. (A.U.)",
                "Shear X (A.U.)",
                "Shear Y (A.U.)",
                "Shear Z (A.U.)",
            ]
            params = pd.DataFrame(params, columns=cols)

        return params

    def params2mats(self, params):
        """ Converts a sequence of global (affine) motion parameters into a sequence
        of 4x4 affine matrices and updates the ``.mat`` attribute. Essentially
        does the opposite of the ``mats2params`` method. 
        
        Parameters
        ----------
        params : np.ndarray
            A 2D numpy array of shape T (time points) x 12
            
        Examples
        --------
        Convert the sequences of affine matrices to a 2D numpy array and uses the
        ``params2mats`` function to reverse it.

        >>> from medusa.data import get_example_h5
        >>> data = get_example_h5(load=True, model="mediapipe")
        >>> orig_mats = data.mat.copy()
        >>> params = data.mats2params(to_df=False)
        >>> data.params2mats(params)
        >>> np.testing.assert_array_almost_equal(orig_mats, data.mat)  # passes!
        """
        T = params.shape[0]
        mats = np.zeros((T, 4, 4))
        for i in range(T):
            p = params[i, :]
            trans, rots, scale, shear = p[:3], p[3:6], p[6:9], p[9:]
            rots = np.deg2rad(rots)
            mats[i, :, :] = compose_matrix(scale, shear, rots, trans)

        self.mat = mats

    def save(self, path, compression_level=9):
        """ Saves (meta)data to disk as an HDF5 file.

        Parameters
        ----------
        path : str
            Path to save the data to
        compression_level : int
            Level of compression (higher = more compression, but slower; max = 9)
        
        Examples
        --------
        Save data to disk:

        >>> import os
        >>> from medusa.data import get_example_h5
        >>> data = get_example_h5(load=True, model="mediapipe")
        >>> data.save('./my_data.h5')
        >>> os.remove('./my_data.h5')  # clean up
        """

        out_dir = Path(path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f_out:

            for attr in ["v", "f", "frame_t", "mat", "cam_mat"]:
                data = getattr(self, attr, None)

                if attr != "f" and data is not None:
                    data = data.astype(np.float32)

                if data is not None:
                    f_out.create_dataset(attr, data=data, compression=compression_level)

            for attr in ["recon_model_name", "img_size", "space"]:
                f_out.attrs[attr] = getattr(self, attr)

            f_out["frame_t"].attrs["sf"] = self.sf
            f_out.attrs["path"] = path
            f_out.attrs["loglevel"] = self.logger.level

        # Note to self: need to do this outside h5py.File context,
        # because Pandas assumes a buffer or path, not an
        if self.events is not None:
            self.events.to_hdf(path, key="events", mode="a")

    @staticmethod
    def load(path):
        """ Loads an HDF5 file from disk, parses its contents, and creates the
        initialization parameters necessary to initialize a ``*Data`` object. It
        does not return a ``*Data`` object itself; only a dictionary with the parameters. 
        
        Important: it is probably better to call the ``load`` method from a specific
        data class (e.g., ``Mediapipe4D``) than the ``load`` method from the
        ``Base4D`` class.
        
        Parameters
        ----------
        path : str, pathlib.Path
            A path towards an HDF5 file data reconstructed by Medusa
            
        Returns
        -------
        init_kwargs : dict
            Parameters necessary to initialize a ``*4D`` object.
            
        Examples
        --------
        Get Mediapipe reconstruction data and initialize a ``Mediapipe4D`` object.
        Note that it's easier to just call the ``load`` classmethod from the
        ``Mediapipe4D`` class directly, i.e., ``Mediapipe4D.load(path)``.

        >>> from medusa.data import get_example_h5
        >>> from medusa.core4d import Mediapipe4D
        >>> path = get_example_h5(load=False, model="mediapipe")
        >>> init_kwargs = Base4D.load(path)
        >>> data = Mediapipe4D(**init_kwargs)               
        """

        init_kwargs = dict()
        with h5py.File(path, "r") as f_in:

            for attr in ["v", "f", "frame_t", "mat", "cam_mat"]:
                if attr in f_in:
                    init_kwargs[attr] = f_in[attr][:]
                else:
                    init_kwargs[attr] = None

            for attr in ["img_size", "recon_model_name", "path", "space", "loglevel"]:
                init_kwargs[attr] = f_in.attrs[attr]

            init_kwargs["sf"] = f_in["frame_t"].attrs["sf"]
            load_events = "events" in f_in

        if load_events:
            init_kwargs["events"] = pd.read_hdf(path, key="events")
        else:
            init_kwargs["events"] = None

        return init_kwargs

    def to_mne_rawarray(self):
        """ Creates an MNE `RawArray` object from the vertices (`v`).
        
        Examples
        --------

        >>> from medusa.data import get_example_h5
        >>> data = get_example_h5(load=True)
        >>> rawarray = data.to_mne_rawarray()
        """
        import mne

        T, nV = self.v.shape[:2]
        info = mne.create_info(
            # vertex 0 (x), vertex 0 (y), vertex 0 (z), vertex 1 (x), etc
            ch_names=[f"v{i}_{c}" for i in range(nV) for c in ["x", "y", "z"]],
            ch_types=["misc"] * np.prod(self.v.shape[1:]),
            sfreq=self.sf,
        )
        return mne.io.RawArray(self.v.reshape(-1, T), info, verbose=False)

    def _rescale(self, img, scaling):
        """ Rescales an image with a scaling factor `scaling`."""
        img = rescale(
            img, scaling, preserve_range=True, anti_aliasing=True, channel_axis=2
        )
        img = img.round().astype(np.uint8)
        return img

    def render_video(
        self, f_out, renderer, video=None, scaling=None, n_frames=None, alpha=None,
        overlay=None
    ):
        """ Renders the sequence of 3D meshes as a video. It is assumed that this
        method is only called from a child class (e.g., ``Mediapipe4D``).
                
        Parameters
        ----------
        f_out: str
            Filename of output
        renderer : ``medusa.render.Renderer``
            The renderer object
        video : str
            Path to video, in order to render face on top of original video frames
        scaling : float
            A scaling factor of the resulting video; 0.25 means 25% of original size
        n_frames : int
            Number of frames to render; e.g., ``10`` means "render only the first
            10 frames of the video"; nice for debugging. If ``None`` (default), all
            frames are rendered
        alpha : float
            Alpha (transparency) level of the rendered face; lower = more transparent;
            minimum = 0 (invisible), maximum = 1 (fully opaque)
        """

        if overlay is not None:
            if overlay.ndim > 2:
                raise ValueError("Overlay should be at most 2 dimensional!")

        w, h = self.img_size
        if scaling is not None:
            w, h = int(round(w * scaling)), int(round(h * scaling))

        if video is not None:
            reader = imageio.get_reader(video)

        writer = imageio.get_writer(f_out, mode="I", fps=self.sf)

        if self.logger.level <= logging.INFO:
            desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")
            iter_ = tqdm(range(len(self)), desc=f"{desc} Render shape")
        else:
            iter_ = range(len(self))

        for i in iter_:

            if n_frames is not None:
                if i == n_frames:
                    break

            if video is not None:
                background = reader.get_data(i)
                if scaling is not None:
                    background = self._rescale(background, scaling)
            else:
                background = np.ones((h, w, 3)).astype(np.uint8) * 255

            if overlay is not None:
                if overlay.ndim == 2:
                    this_overlay = overlay[i, :]
                else:
                    this_overlay = overlay
            else:
                this_overlay = None

            img = renderer(self.v[i, :, :], self.f, this_overlay)
            if scaling is not None:
                img = self._rescale(img, scaling)

            img = renderer.alpha_blend(img, background, face_alpha=alpha)
            writer.append_data(img)

        renderer.close()
        writer.close()

        if video is not None:
            reader.close()

    def plot_data(self, f_out, plot_motion=True, plot_pca=True, n_pca=3):
        """ Creates a plot of the motion (rotation & translation) parameters
        over time and the first `n_pca` PCA components of the
        reconstructed vertices. For FLAME and Mediapipe estimates, these parameters are
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
        
        Examples
        --------

        >>> import os
        >>> from medusa.data import get_example_h5
        >>> data = get_example_h5(load=True)
        >>> data.plot_data('./example_plot.png')
        >>> os.remove('./example_plot.png')    
        """

        if self.mat is None:
            self.logger.warn("No motion params available; setting plot_motion to False")
            plot_motion = False

        if not plot_motion and not plot_pca:
            raise ValueError("`plot_motion` and `plot_pca` cannot both be False")

        if plot_motion and plot_pca:
            nrows = 13
        elif plot_motion:
            nrows = 12
        else:
            nrows = 1

        fig, axes = plt.subplots(nrows=nrows, sharex=True, figsize=(10, 2 * nrows))
        if nrows == 1:
            axes = [axes]

        t = self.frame_t - self.frame_t[0]  # time in sec.

        if plot_motion:
            motion = self.mats2params()
            for i, col in enumerate(motion.columns):
                axes[i].plot(t, motion[col] - motion.loc[0, col])
                axes[i].set_ylabel(col, fontsize=10)

        if plot_pca:
            # Perform PCA on flattened vertex array (T x (nV x 3))
            pca = PCA(n_components=n_pca)
            v_pca = pca.fit_transform(self.v.reshape(self.v.shape[0], -1))
            axes[-1].plot(t, v_pca)
            axes[-1].legend([f"PC{i+1}" for i in range(n_pca)])
            axes[-1].set_ylabel("Movement (A.U.)", fontsize=10)

        for ax in axes:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.grid(True)

            if self.events is not None:
                # Also plot a dashed vertical line for each trial (with a separate
                # color for each condition)
                for i, tt in enumerate(self.events["trial_type"].unique()):
                    ev = self.events.query("trial_type == @tt")
                    for _, ev_i in ev.iterrows():
                        onset = ev_i["onset"] - self.frame_t[0]
                        ax.axvline(onset, ls="--", c=plt.cm.Set1(i))

        axes[-1].set_xlabel("Time (sec.)", fontsize=10)
        fig.savefig(f_out)
        plt.close()

    def __len__(self):
        """ Returns the number of time points of the reconstructed vertices (i.e.,
        the number of reconstructed frames from the video. """
        return self.v.shape[0]

    def __getitem__(self, idx):
        """ Returns the vertices at a particular time point (``idx``).
        
        Parameters
        ----------
        idx : int
            Index into the time dimension of the data
        """
        return self.v[idx, :, :]

    def __setitem__(self, idx, v):
        """ Replace the vertices at time point ``idx`` with ``v``. 
        
        Parameters
        ----------
        idx : int
            Index into the time dimension of the data
        v : np.ndarray
            Numpy array with vertices of shape ``nV`` (number of verts) x 3 (XYZ)
        """
        self.v[idx, ...] = v


class Flame4D(Base4D):
    """4D data class specific to reconstructions from models based on the FLAME
    topology.

    Warning: we recommend against initializing a ``Flame4D`` object directly
    (i.e., through the ``__init__`` class constructor). Instead, use the high-level
    ``videorecon`` function, which returns a ``Flame4D`` object. Or, if you
    are loading data from disk, use the ``load`` classmethod (see examples)

    Parameters
    ----------
    *args : iterable
        Positional (non-keyword) arguments passed to the ``Base4D`` constructor
    **kwargs: dict
        Keyword arguments passed to the ``Base4D`` constructor
        
    Examples
    --------
    We recommend creating ``Flame4D`` objects by loading the corresponding
    HDF5 file from disk (see ``load`` docstring). 
    """
    
    def __init__(self, *args, **kwargs):    

        if kwargs.get("cam_mat") is None:
            cam_mat = np.eye(4)
            cam_mat[2, 3] = 4  # zoom out 4 units in z direction
            kwargs["cam_mat"] = cam_mat

        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path):
        """ Loads existing data (stored as an HDF5 file) from disk and uses it to
        instantiate a ``Flame4D`` object. 
        
        Parameters
        ----------
        path : str, pathlib.Path
            A path to an HDF5 file with data from a Flame-based reconstruction model
        
        Returns
        -------
        An ``Flame4D`` object
        
        Examples
        --------
        Load data from a ``mediapipe`` reconstruction:
        
        >>> from medusa.data import get_example_h5
        >>> path_to_h5 = get_example_h5(load=False)
        >>> data = Flame4D.load(path_to_h5)
        >>> type(data)
        <class 'medusa.core4d.Flame4D'>
        """

        # Note to self: cannot use the super().load method as a classmethod, because
        # then we don't have access to the ``Flame4D`` cls
        init_kwargs = super().load(path)
        return cls(**init_kwargs)

    def render_video(self, f_out, smooth=False, wireframe=False, **kwargs):
        """ Renders a video from the 4D reconstruction.
        
        Parameters
        ----------
        f_out : str, pathlib.Path
            Path to save the video to
        smooth : bool
            Whether to render a smooth face (using smooth shading) or not (using flat
            shading)
        wireframe : bool
            Whether to render a wireframe instead of an opaque face (if ``True``, the
            ``smooth`` parameter is ignored)
        kwargs : dict
            Additional keyword arguments passed to the ``Base4D.render_video`` method
        
        Examples
        --------
        Render a video 
        
        """ 
        renderer = Renderer(
            camera_type="orthographic",
            viewport=self.img_size,
            smooth=smooth,
            wireframe=wireframe,
            cam_mat=self.cam_mat,
        )

        super().render_video(f_out, renderer, **kwargs)


class Mediapipe4D(Base4D):
    """4D data class specific to reconstructions from the Mediapipe model.

    Warning: we recommend against initializing a ``Mediapipe4D`` object directly
    (i.e., through the ``__init__`` class constructor). Instead, use the high-level
    ``videorecon`` function, which returns a ``Mediapipe4D`` object. Or, if you
    are loading data from disk, use the ``load`` classmethod (see examples)

    Parameters
    ----------
    *args : iterable
        Positional (non-keyword) arguments passed to the ``Base4D`` constructor
    **kwargs: dict
        Keyword arguments passed to the ``Base4D`` constructor
        
    Examples
    --------
    We recommend creating ``Mediapipe4D`` objects by loading the corresponding
    HDF5 file from disk (see ``load`` docstring). 
    """
    def __init__(self, *args, **kwargs):
        #kwargs["f"] = get_template_mediapipe()['f']
        if kwargs.get("cam_mat") is None:
            kwargs["cam_mat"] = np.eye(4)

        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path):
        """ Loads Mediapipe data from a HDF5 file and returns a ``Mediapipe4D``
        object.
        
        Parameters
        ----------
        path : str, pathlib.Path
            Path to HDF5 file with Mediapipe data
        
        Returns
        -------
        A ``Mediapipe4D`` object
        
        Examples
        --------
        The ``load`` classmethod is the recommended way to initialize a ``Mediapipe4D``
        object with already reconstructed data:
        
        >>> from medusa.data import get_example_h5
        >>> path = get_example_h5()
        >>> mp_data = Mediapipe4D.load(path)

        If the data is not reconstructed yet, use the ``videorecon`` function to create
        such an object:
        
        >>> from medusa.preproc import videorecon
        >>> from medusa.data import get_example_video
        >>> path = get_example_video()
        >>> mp_data = videorecon(path, recon_model_name='mediapipe')
        """
        init_kwargs = super().load(path)
        return cls(**init_kwargs)

    def render_video(self, f_out, smooth=False, wireframe=False, **kwargs):
        """ Renders a video of the reconstructed vertices. 
        
        Note: the extension of the ``f_out`` parameter (e.g., ".gif" or ".mp4")
        determines the format of the rendered video.
        
        Parameters
        ----------
        f_out : str, pathlib.Path
            Path where the video should be saved
        smooth : bool
            Whether to render a smooth mesh or not (ignored when ``wireframe=True``)
        wireframe : bool
            Whether to render wireframe instead of the full mesh
        **kwargs : dict
            Keyword arguments passed to the ``render_video`` method from ``Base4D``;
            options include ``video``, ``scaling``, ``n_frames``, and ``alpha``
            
        Examples
        --------
        Rendering a GIF with just the wireframe:
        
        >>> from pathlib import Path
        >>> from medusa.data import get_example_h5
        >>> data = get_example_h5(load=True)
        >>> f_out = Path('./example_vid_recon.gif')
        >>> data.render_video(f_out, wireframe=True)
        >>> f_out.is_file()
        True

        Rendering an MP4 video with a smooth mesh on top of the original video:
        
        >>> from medusa.data import get_example_video 
        >>> vid = get_example_video()
        >>> data = get_example_h5(load=True)
        >>> f_out = Path('./example_vid_recon.mp4')
        >>> data.render_video(f_out, smooth=True, video=vid)
        >>> f_out.is_file()
        True
        """
        renderer = Renderer(
            camera_type="intrinsic",
            viewport=self.img_size,
            smooth=smooth,
            wireframe=wireframe,
            cam_mat=self.cam_mat,
        )
        super().render_video(f_out, renderer, **kwargs)


class Fan4D(Base4D):
    """Data class specific to reconstructions from the FAN (3D) model.

    Warning: we recommend against initializing a ``Fan4D`` object directly
    (i.e., through the ``__init__`` class constructor). Instead, use the high-level
    ``videorecon`` function, which returns a ``Fan4D`` object. Or, if you
    are loading data from disk, use the ``load`` classmethod (see examples)

    Parameters
    ----------
    *args : iterable
        Positional (non-keyword) arguments passed to the ``Base4D`` constructor
    **kwargs: dict
        Keyword arguments passed to the ``Base4D`` constructor
        
    Examples
    --------
    We recommend creating ``Fan4D`` objects by loading the corresponding
    HDF5 file from disk (see ``load`` docstring). 
    """

    def __init__(self, *args, **kwargs):
        here = Path(__file__).parent.resolve()
        kwargs["f"] = np.load(here / "data/faces_fan.npy")
        kwargs["cam_mat"] = np.eye(4)
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path):
        """ Loads FAN data from a HDF5 file and returns a ``Fan4D`` object.
        
        Parameters
        ----------
        path : str, pathlib.Path
            Path to HDF5 file with FAN data
        
        Returns
        -------
        A ``Fan4D`` object
        
        Examples
        --------
        If the data is not reconstructed yet, use the ``videorecon`` function to create
        such an object:
        
        >>> from medusa.preproc import videorecon
        >>> from medusa.data import get_example_video
        >>> path = get_example_video()
        >>> fan_data = videorecon(path, recon_model_name='fan', device='cpu')
        """

        init_kwargs = super().load(path)
        return cls(**init_kwargs)

    def render_video(self, f_out, video=None, scaling=None, n_frames=None, **kwargs):
        """ Renders a video of the reconstructed vertices. 
        
        Note: the extension of the ``f_out`` parameter (e.g., ".gif" or ".mp4")
        determines the format of the rendered video.
        
        Parameters
        ----------
        f_out : str, pathlib.Path
            Path where the video should be saved
        video : str, pathlib.Path
            Path to video, if you want to render the face on top of the original video;
            default is ``None`` (i.e., do not render on top of video)
            
        Examples
        --------
        Rendering a GIF with wireframe (only possibility) on top of video:
        
        >>> from pathlib import Path
        >>> from medusa.data import get_example_video
        >>> from medusa.data import get_example_h5
        >>> vid = get_example_video()
        >>> data = get_example_h5(load=True, model='fan')
        >>> f_out = Path('./example_vid_recon.gif')
        >>> data.render_video(f_out, video=vid)
        >>> f_out.is_file()
        True
        """

        if video is not None:
            # Plot face on top of video, so need to load in video
            reader = imageio.get_reader(video)
        
        v = self.v

        w, h = self.img_size
        if scaling is not None:
            w, h = int(round(w * scaling)), int(round(h * scaling))

        writer = imageio.get_writer(f_out, mode="I", fps=self.sf)
        desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")
        
        if self.logger.level <= logging.INFO:
            iter_ = tqdm(range(len(self)), desc=f"{desc} Render shape")
        else:
            iter_ = range(len(self))

        for i in iter_:

            if n_frames is not None:
                if i == n_frames:
                    break

            if video is not None:
                background = reader.get_data(i)
                if scaling is not None:
                    background = self._rescale(background, scaling)    
            else:
                background = np.zeros((w, h, 3)).astype(np.uint8)

            this_v = v[i, ...]
            if scaling is not None:
                this_v = np.round(this_v * scaling).astype(int)
            else:
                this_v = np.round(this_v).astype(int)

            for ii in range(self.v.shape[1]):
                cv2.circle(
                    background,
                    this_v[ii, :2],
                    radius=1,
                    color=(255, 0, 0),
                    thickness=1,
                )

            lines = [(i, i+1) for i in range(16)]  # face contour
            lines.extend([[i, i+1] for i in range(17, 21)])  # right eyebrow
            lines.extend([[i, i+1] for i in range(22, 26)])  # left eyebrow
            lines.extend([[i, i+1] for i in range(27, 30)])  # nose ridge
            lines.extend([[i, i+1] for i in range(31, 35)])  # nose arc
            lines.extend([[i, i+1] for i in range(36, 41)] + [[41, 36]])  # right eye
            lines.extend([[i, i+1] for i in range(42, 47)] + [[47, 42]])  # left eye
            lines.extend([[i, i+1] for i in range(48, 59)] + [[59, 48]])  # lip outer
            lines.extend([[i, i+1] for i in range(60, 67)] + [[67, 60]])  # lip inner
            
            for line in lines:
                cv2.line(background,
                         this_v[line[0], :2], this_v[line[1], :2], (255, 0, 0),
                         thickness=1)

            writer.append_data(background)

        writer.close()
        if video is not None:
            reader.close()

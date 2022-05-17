import os

os.environ["DISPLAY"] = ":0.0"

import cv2
import h5py
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


class BaseData:
    """Base Data class with attributes and methods common to all Data
    classes (such as FlameData, MediapipeData, etc.).

    Warning: objects should never be initialized with this class directly,
    only when calling super().__init__() from the subclass (like `FlameData`).

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
    """

    def __init__(
        self,
        v,
        f=None,
        mat=None,
        cam_mat=None,
        frame_t=None,
        events=None,
        sf=None,
        img_size=None,
        recon_model_name=None,
        space="world",
        path=None,
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

        # if self.cam_mat is None:
        #    logger.info("Setting camera matrix to identity")
        #    self.cam_mat = np.eye(4)

    def mats2params(self, to_df=True):
        """Transforms a time series (of length T) 4x4 affine matrices to a
        pandas DataFrame with a time series of T x 12 affine parameters
        (translation XYZ, rotation XYZ, scale XYZ, shear XYZ).
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
        """Does the oppose as the above function."""
        T = params.shape[0]
        mats = np.zeros((T, 4, 4))
        for i in range(T):
            p = params[i, :]
            trans, rots, scale, shear = p[:3], p[3:6], p[6:9], p[9:]
            rots = np.deg2rad(rots)
            mats[i, :, :] = compose_matrix(scale, shear, rots, trans)

        self.mat = mats

    def save(self, path, compression_level=9):
        """Saves data to disk as a hdf5 file.

        Parameters
        ----------
        path : str
            Path to save the data to
        """

        out_dir = Path(path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f_out:

            for attr in ["v", "f", "frame_t", "mat", "cam_mat"]:
                data = getattr(self, attr, None)
                if attr != "f":
                    data = data.astype(np.float32)

                if data is not None:
                    f_out.create_dataset(attr, data=data, compression=compression_level)

            for attr in ["recon_model_name", "img_size", "space"]:
                f_out.attrs[attr] = getattr(self, attr)

            f_out["frame_t"].attrs["sf"] = self.sf
            f_out.attrs["data_class"] = self.__class__.__name__
            f_out.attrs["path"] = path

        # Note to self: need to do this outside h5py.File context,
        # because Pandas assumes a buffer or path, not an
        if self.events is not None:
            self.events.to_hdf(path, key="events", mode="a")

    @staticmethod
    def load(path):
        """Loads a hdf5 file from disk and returns a Data object."""

        init_kwargs = dict()
        with h5py.File(path, "r") as f_in:

            for attr in ["v", "f", "frame_t", "mat", "cam_mat"]:
                if attr in f_in:
                    init_kwargs[attr] = f_in[attr][:]
                else:
                    init_kwargs[attr] = None

            for attr in ["img_size", "recon_model_name", "path", "space"]:
                init_kwargs[attr] = f_in.attrs[attr]

            init_kwargs["sf"] = f_in["frame_t"].attrs["sf"]
            load_events = "events" in f_in

        if load_events:
            init_kwargs["events"] = pd.read_hdf(path, key="events")
        else:
            init_kwargs["events"] = None

        return init_kwargs

    def events_to_mne(self):
        """Converts events DataFrame to (N x 3) array that
        MNE expects.

        Returns
        -------
        events : np.ndarray
            An N (number of trials) x 3 array, with the first column
            indicating the sample *number* indicating the
        """

        if self.events is None:
            raise ValueError("There are no events associated with this data object!")

        event_id = {k: i for i, k in enumerate(self.events["trial_type"].unique())}
        events = np.zeros((self.events.shape[0], 3))
        for i, (_, ev) in enumerate(self.events.iterrows()):
            events[i, 2] = event_id[ev["trial_type"]]
            events[i, 0] = np.argmin(np.abs(self.frame_t - ev["onset"]))

            if events[i, 0] > 0.05:
                raise ValueError(
                    f"Nearest sample > 0.05 seconds away for trial {i+1}; "
                    "Try resampling the data to a higher resolution!"
                )

        return events, event_id

    def to_mne_rawarray(self):
        """Creates an MNE `RawArray` object from the vertices (`v`)."""
        import mne

        T, nV = self.v.shape[:2]
        info = mne.create_info(
            # vertex 0 (x), vertex 0 (y), vertex 0 (z), vertex 1 (x), etc
            ch_names=[f"v{i}_{c}" for i in range(nV) for c in ["x", "y", "z"]],
            ch_types=["misc"] * np.prod(self.v.shape[1:]),
            sfreq=self.sf,
        )
        return mne.io.RawArray(self.v.reshape(T, -1), info)

    def _rescale(self, img, scaling):
        """Rescales an image with a scaling factor `scaling`."""
        img = rescale(
            img, scaling, preserve_range=True, anti_aliasing=True, channel_axis=2
        )
        img = img.round().astype(np.uint8)
        return img

    def render_video(
        self, f_out, renderer, video=None, scaling=None, n_frames=None, alpha=None
    ):
        """Should be implemented in subclass!"""

        w, h = self.img_size
        if scaling is not None:
            w, h = int(round(w * scaling)), int(round(h * scaling))

        if video is not None:
            reader = imageio.get_reader(video)

        writer = imageio.get_writer(f_out, mode="I", fps=self.sf)
        desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")

        for i in tqdm(range(len(self)), desc=f"{desc} Render shape"):

            if n_frames is not None:
                if i == n_frames:
                    break

            if video is not None:
                background = reader.get_data(i)
                if scaling is not None:
                    background = self._rescale(background, scaling)
            else:
                background = np.zeros((h, w, 3)).astype(np.uint8)

            img = renderer(self.v[i, :, :], self.f)
            if scaling is not None:
                img = self._rescale(img, scaling)

            img = renderer.alpha_blend(img, background, face_alpha=alpha)
            writer.append_data(img)

        renderer.close()
        writer.close()

        if video is not None:
            reader.close()

    def plot_data(self, f_out, plot_motion=True, plot_pca=True, n_pca=3):
        """Creates a plot of the motion (rotation & translation) parameters
        over time and the first `n_pca` PCA components of the
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
        return self.v.shape[0]

    def __getitem__(self, idx):
        return self.v[idx, :, :]

    def __setitem__(self, idx, v):
        self.v[idx, ...] = v


class FlameData(BaseData):
    def __init__(self, *args, **kwargs):
        here = Path(__file__).parent.resolve()
        kwargs["f"] = np.load(here / "data/faces_flame.npy")
        if kwargs.get("cam_mat") is None:
            cam_mat = np.eye(4)
            cam_mat[2, 3] = 4  # zoom out 4 units in z direction
            kwargs["cam_mat"] = cam_mat

        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path):
        here = Path(__file__).parent.resolve()
        init_kwargs = super().load(path)
        init_kwargs["f"] = np.load(here / "data/faces_flame.npy")
        return cls(**init_kwargs)

    def render_video(self, f_out, smooth=False, wireframe=False, **kwargs):

        renderer = Renderer(
            camera_type="orthographic",
            viewport=self.img_size,
            smooth=smooth,
            wireframe=wireframe,
            cam_mat=self.cam_mat,
        )

        super().render_video(f_out, renderer, **kwargs)


class MediapipeData(BaseData):
    def __init__(self, *args, **kwargs):
        here = Path(__file__).parent.resolve()
        kwargs["f"] = np.load(here / "data/faces_mediapipe.npy")
        if kwargs.get("cam_mat") is None:
            kwargs["cam_mat"] = np.eye(4)

        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path):

        init_kwargs = super().load(path)
        return cls(**init_kwargs)

    def render_video(self, f_out, smooth=False, wireframe=False, **kwargs):

        renderer = Renderer(
            camera_type="intrinsic",
            viewport=self.img_size,
            smooth=smooth,
            wireframe=wireframe,
            cam_mat=self.cam_mat,
        )
        super().render_video(f_out, renderer, **kwargs)


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

        writer = imageio.get_writer(f_out, mode="I", fps=self.sf)
        desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")

        for i in tqdm(range(len(self)), desc=f"{desc} Render shape"):

            # fig, ax = plt.subplots()
            if video is not None:
                background = reader.get_data(i)
            else:
                background = np.zeros((w, h, 3)).astype(np.uint8)

            for ii in range(self.v.shape[1]):
                cv2.circle(
                    background,
                    np.round(v[i, ii, :2]).astype(int),
                    radius=2,
                    color=(255, 0, 0),
                    thickness=3,
                )

            writer.append_data(background)

        writer.close()
        if video is not None:
            reader.close()


MODEL2CLS = {"emoca": FlameData, "mediapipe": MediapipeData, "FAN-3D": FANData}


def load_h5(path):
    """Convenience function to load a hdf5 file and
    immediately initialize the correct data class.

    Located here (instead of io.py or render.py) to
    prevent circular imports.

    Parameters
    ----------
    path : str
        Path to hdf5 file

    Returns
    -------
    data : data.BaseData subclass object
        An object with a class derived from data.BaseData
        (like MediapipeData, or FlameData)
    """
    # peek at recon model
    with h5py.File(path, "r") as f_in:
        rmn = f_in.attrs["recon_model_name"]

    data = MODEL2CLS[rmn].load(path)
    return data

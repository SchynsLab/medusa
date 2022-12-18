"""Module with core 4D functionality of the ``medusa`` package, most
importantly the ``*4D`` classes. The ``Base4D`` class defines a template class
from which model-specific classes (such as ``Flame4D``) inherit. Objects
initialized from these classes store reconstructed data from videos and other
(meta)data needed to further process, analyze, and visualize it.

The reconstructed data from each model supported by ``medusa`` is stored in an object from
a specific class which inherits from ``Base4D``. For example, reconstructed data from
`mediapipe <https://google.github.io/mediapipe/solutions/face_mesh.html>`_ is stored
in using the ``Mediapipe4D`` class. Other classes include the ``Flame4D`` for reconstructions
from models using the `FLAME topology <https://flame.is.tue.mpg.de/>`_ (such as
`EMOCA <https://emoca.is.tue.mpg.de/>`_).

The data can be saved to disk as a `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
file (using `h5py <http://www.h5py.org/>`_) with the ``save`` method and loaded from
disk using the ``load`` (static)method.
"""
import torch
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from skimage.transform import rescale
from tqdm import tqdm
from trimesh import Trimesh
from trimesh.transformations import compose_matrix, decompose_matrix

from .. import DEVICE
from ..io import VideoWriter, VideoLoader
from ..log import get_logger, tqdm_log
from ..render import Renderer


class Data4D:
    """Base Data class with attributes and methods common to all 4D data
    classes (such as ``Flame4D``, ``Mediapipe4D``, etc.).

    Warning: objects should never be initialized with this class directly,
    only when calling ``super().__init__()`` from the subclass (like ``Flame4D``). Note,
    though, that the initialization parameters are the same for every class that
    inherits from ``Base4D``.

    Parameters
    ----------
    v : ndarray
        Numpy array of shape T (time points) x nV (no. vertices) x 3 (x/y/z)
    tris : ndarray
        Integer numpy array of shape n_t (no. of triangles) x 3 (vertices per triangle)
    face_idx : ndarray
        Integer numpy array with indices that map vertices to distinct faces
    mat : ndarray
        Numpy array of shape T (time points) x 4 x 4 (affine matrix) representing
        the 'world' (or 'model') matrix for each time point
    cam_mat : ndarray
        Numpy array of shape 4x4 (affine matrix) representing the camera matrix
    frame_t : ndarray
        Numpy array of length T (time points) with "frame times", i.e.,
        onset of each frame (in seconds) from the video
    sf : int, float
        Sampling frequency of video
    recon_model : str
        Name of reconstruction model
    space : str
        The space the vertices are currently in; can be either 'local' or 'world'
    path : str
        Path where the data is saved; if initializing a new object (rather than
        loading one from disk), this should be `None`
    loglevel : int
        Logging level of current logger
    """

    def __init__(self, v, mat, tris, img_idx=None, face_idx=None, video_metadata=None,
                 space="world", device=DEVICE, loglevel='INFO'):

        self.v = v
        self.mat = mat
        self.tris = tris
        self.img_idx = img_idx
        self.face_idx = face_idx
        self.video_metadata = video_metadata
        self.space = space
        self.device = device
        self.logger = get_logger(loglevel)
        self._check()

    def _check(self):
        """Does some checks to make sure the data works with the renderer and
        other stuff."""

        # Renderer expects torch floats (float32), not double (float64)
        if self.v.dtype != torch.float32:
            self.v = self.v.float()

        if self.img_idx is None:
            self.img_idx = torch.arange(self.v.shape[0], device=self.device)

        if self.face_idx is None:
            self.face_idx = torch.zeros(self.v.shape[0], device=self.device)

        T = self.v.shape[0]
        if self.mat.shape[0] != T:
            mT = self.mat.shape[0]
            self.logger.warning(f"More mats ({mT}) than vertex time points ({T}); "
                                 "trimming ...")
            self.mat = self.mat[:T, :, :]

        if self.space not in ["local", "world"]:
            raise ValueError("`space` should be either 'local' or 'world'!")

    def project_to_68_landmarks(self):
        """Projects to 68 landmark set."""

        if self.v.shape[1:] == (468, 3):
            fname = 'mediapipe_lmk68_embedding.npz'
        else:
            fname = 'flame_lmk68_embedding.npz'

        emb = np.load(Path(__file__).parents[1] / f'data/{fname}')
        vf = self.v[:, self.f[emb['lmk_faces_idx']]]  # T x V x 3 (faces) x 3 (faces) x 3 (xyz)
        v_proj = np.sum(vf * emb['lmk_bary_coords'][:, :, None], axis=2)

        return v_proj

    def decompose_mats(self, to_df=True):
        """Decomponses a time series (of length T) 4x4 affine matrices to a
        numpy array (or pandas ``DataFrame``) with a time series of T x 12
        affine parameters (translation XYZ, rotation XYZ, scale XYZ, shear
        XYZ).

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
        >>> params = data.decompose_mats(to_df=False)
        >>> params.shape
        (232, 12)
        """

        if self.mat is None:
            raise ValueError("Cannot convert matrices to parameters because "
                             "there are no matrices (self.mat is None)!")

        T = self.mat.shape[0]
        params = np.zeros((T, 12))
        for i in range(T):
            scale, shear, angles, trans, _ = decompose_matrix(self.mat[i, :, :])
            params[i, :3] = trans
            params[i, 3:6] = np.rad2deg(angles)
            params[i, 6:9] = scale
            params[i, 9:12] = shear

        if to_df:
            cols = ["Trans. X", "Trans. Y", "Trans. Z",
                    "Rot. X (deg)", "Rot. Y (deg)", "Rot. Z (deg)",
                    "Scale X (A.U.)", "Scale Y (A.U.)", "Scale Z. (A.U.)",
                    "Shear X (A.U.)", "Shear Y (A.U.)", "Shear Z (A.U.)"]

            params = pd.DataFrame(params, columns=cols)

        return params

    def compose_mats(self, params):
        """Converts a sequence of global (affine) motion parameters into a
        sequence of 4x4 affine matrices and updates the ``.mat`` attribute.
        Essentially does the opposite of the ``decompose_mats`` method.

        Parameters
        ----------
        params : np.ndarray
            A 2D numpy array of shape T (time points) x 12

        Examples
        --------
        Convert the sequences of affine matrices to a 2D numpy array and uses the
        ``compose_mats`` function to reverse it.

        >>> from medusa.data import get_example_h5
        >>> data = get_example_h5(load=True, model="mediapipe")
        >>> orig_mats = data.mat.copy()
        >>> params = data.decompose_mats(to_df=False)
        >>> data.compose_mats(params)
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

    def save_obj(self, idx, path):

        with open(path, 'w') as f_out:
            Trimesh(self.v[idx, ...], self.tris).export(f_out, file_type='obj')

    def save(self, path, compression_level=9):
        """Saves (meta)data to disk as an HDF5 file.

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

        if not isinstance(path, Path):
            path = Path(path)

        out_dir = path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f_out:

            for attr, data in self.__dict__.items():

                if attr == 'logger':
                    data = data.getEffectiveLevel()

                if torch.is_tensor(data):
                    data = data.cpu().numpy()

                if isinstance(data, np.ndarray):
                    f_out.create_dataset(attr, data=data, compression=compression_level)
                elif isinstance(data, dict):
                    f_out.create_group(attr)
                    f_out[attr].attrs.update(data)
                else:
                    f_out.attrs[attr] = data

    @classmethod
    def load(cls, path, device=None):
        """Loads an HDF5 file from disk, parses its contents, and creates the
        initialization parameters necessary to initialize a ``*Data`` object.
        It does not return a ``*Data`` object itself; only a dictionary with
        the parameters.

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
        >>> from medusa.core import Mediapipe4D
        >>> path = get_example_h5(load=False, model="mediapipe")
        >>> init_kwargs = Base4D.load(path)
        >>> data = Mediapipe4D(**init_kwargs)
        """

        init_kwargs = dict()
        with h5py.File(path, "r") as f_in:

            if device is None:
                device = f_in.attrs.get('device', DEVICE)

            for attr, data in f_in.items():
                if isinstance(data, h5py.Group):
                    data = dict(data.attrs)
                elif isinstance(data, h5py.Dataset):
                    data = torch.as_tensor(data[:], device=device)

                init_kwargs[attr] = data

            for attr, value in f_in.attrs.items():
                init_kwargs[attr] = value

        init_kwargs['loglevel'] = init_kwargs.pop('logger')

        return cls(**init_kwargs)

    def _rescale(self, img, scale):
        """Rescales an image with a scaling factor `scaling`."""
        img = rescale(img, scale, preserve_range=True, anti_aliasing=True, channel_axis=2)
        img = img.round().astype(np.uint8)
        return img

    def render_video(self, f_out, video=None, alpha=1, **kwargs):
        """Renders the sequence of 3D meshes as a video. It is assumed that
        this method is only called from a child class (e.g., ``Mediapipe4D``).

        Parameters
        ----------
        f_out: str
            Filename of output
        renderer : ``medusa.render.Renderer``
            The renderer object
        video : str
            Path to video, in order to render face on top of original video frames
        scale : float
            A scaling factor of the resulting video; 0.25 means 25% of original size
        n_frames : int
            Number of frames to render; e.g., ``10`` means "render only the first
            10 frames of the video"; nice for debugging. If ``None`` (default), all
            frames are rendered
        alpha : float
            Alpha (transparency) level of the rendered face; lower = more transparent;
            minimum = 0 (invisible), maximum = 1 (fully opaque)
        """

        cam_mat = np.eye(4)
        if self.v.shape[1] == 468:
            cam_type = 'intrinsic'
        else:
            cam_type = 'orthographic'
            cam_mat[2, 3] = 4  # zoom out 4 units in z direction

        w, h = self.video_metadata['img_size']
        renderer = Renderer(viewport=(w, h), cam_mat=cam_mat,
                            cam_type=cam_type, **kwargs)

        if video is not None:
            reader = VideoLoader(video, batch_size=1, device='cpu')

        writer = VideoWriter(str(f_out), fps=self.video_metadata['fps'])
        iter_ = tqdm_log(range(self.v.shape[0]), self.logger, desc='Render shape')

        for i in iter_:

            if video is not None:
                background = next(iter(reader)).numpy()
            else:
                background = np.ones((h, w, 3)).astype(np.uint8) * 255

            img = renderer(self.v[i, :, :], self.tris)
            img = renderer.alpha_blend(img, background, face_alpha=alpha)

            writer.write(img)

        renderer.close()
        writer.close()

        if video is not None:
            reader.close()

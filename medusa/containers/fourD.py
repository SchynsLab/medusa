"""Module with core 4D functionality of the ``medusa`` package, most
importantly the ``Data4D`` class, which stores reconstructed data from videos
and other (meta)data needed to further process, analyze, and visualize it.

The data can be saved to disk as a `HDF5
<https://www.hdfgroup.org/solutions/hdf5/>`_ file (using `h5py
<http://www.h5py.org/>`_) with the ``save`` method and loaded from disk
using the ``load`` classmethod.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from kornia.geometry.linalg import transform_points

from ..defaults import DEVICE, LOGGER
from ..io import VideoLoader, VideoWriter
from ..log import tqdm_log
from ..tracking import filter_faces, _ensure_consecutive_face_idx
from ..transforms import compose_matrix, decompose_matrix


class Data4D:
    """Data class which stores reconstruction data and provides methods to
    preprocess/manipulate them.

    Parameters
    ----------
    v : np.ndarray, torch.tensor
        Numpy array or torch tensor of shape T (time points) x nV (no. vertices) x 3 (x/y/z)
    tris : ndarray, torch.tensor
        Integer numpy array or torch tensor of shape n_t (no. of triangles) x 3 (vertices per triangle)
    mat : ndarray
        Numpy array of shape T (time points) x 4 x 4 (affine matrix) representing
        the 'world' (or 'model') matrix for each time point
    face_idx : ndarray
        Integer numpy array with indices that map vertices to distinct faces
    cam_mat : ndarray
        Numpy array of shape 4x4 (affine matrix) representing the camera matrix
    space : str
        The space the vertices are currently in; can be either 'local' or 'world'
    """

    def __init__(self, v, mat, tris=None, img_idx=None, face_idx=None,
                 video_metadata=None, cam_mat=None, space="world", device=DEVICE):
        """Initializes a Data4D object."""
        self.v = v
        self.mat = mat
        self.tris = tris
        self.img_idx = img_idx
        self.face_idx = face_idx
        self.video_metadata = video_metadata
        self.cam_mat = cam_mat
        self.space = space
        self.device = device
        self._check()

    def _check(self):
        """Does some checks to make sure the data works with the renderer and
        other stuff."""

        if self.mat.ndim == 2 and self.mat.shape[1] == 12:
            # probably dealing with params instead of mats
            self.compose_mats(self.mat)

        B, V, _ = self.v.shape  # batch size, number of vertices
        if self.img_idx is None:
            self.img_idx = torch.arange(B, dtype=torch.int64, device=self.device)

        if self.face_idx is None:
            self.face_idx = torch.zeros(B, dtype=torch.int64, device=self.device)

        if self.tris is None:
            from ..data import get_tris  # avoids circular import
            self.tris = get_tris(self._infer_topo(), self.device)

        for attr in ('v', 'mat', 'tris', 'img_idx', 'face_idx', 'cam_mat'):
            data = getattr(self, attr, None)
            if isinstance(data, np.ndarray):
                data = torch.as_tensor(data, device=self.device)
                setattr(self, attr, data)

        if self.video_metadata is None:
            self.video_metadata = {
                'img_size': None,
                'n_img': self.v.shape[0],
                'fps': 30
            }

        if self.cam_mat is None:
            self.cam_mat = torch.eye(4, device=self.device)

        for attr in ('v', 'mat', 'tris', 'face_idx', 'img_idx', 'cam_mat'):
            data = getattr(self, attr)
            if data.device.type != self.device:
                data = data.to(self.device)

            if attr in ('v', 'mat', 'cam_mat'):
                data = data.to(torch.float32)
            else:
                data = data.to(torch.int64)

            setattr(self, attr, data)

        if self.space not in ["local", "world"]:
            raise ValueError("`space` should be either 'local' or 'world'!")

    def _infer_topo(self):
        """Tries to infer the topology of the current vertices."""
        nv = self.v.shape[1]
        if nv == 468:
            return 'mediapipe'
        elif nv == 59315:
            return 'flame-dense'
        else:
            # Could be that mask is applied, but not an ideal situation here; must be
            # another way to check which topo we're dealing with
            return 'flame-coarse'

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
        >>> from medusa.data import get_example_data4d
        >>> data = get_example_data4d(load=True, model="mediapipe")
        >>> data.save('./my_data.h5')
        >>> os.remove('./my_data.h5')  # clean up
        """

        if not isinstance(path, Path):
            path = Path(path)

        out_dir = path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f_out:

            for attr, data in self.__dict__.items():

                if attr[0] == '_':
                    continue

                if torch.is_tensor(data):
                    data = data.cpu().numpy()

                if isinstance(data, np.ndarray):
                    f_out.create_dataset(attr, data=data, compression=compression_level)
                elif isinstance(data, dict):
                    f_out.create_group(attr)
                    f_out[attr].attrs.update(data)
                else:
                    f_out.attrs[attr] = data

    def apply_vertex_mask(self, name):
        """Applies a mask to the vertices (and triangles).

        Parameters
        ----------
        name : str
            Name of masks (one of 'face', 'lips', 'neck', 'nose', 'boundary', 'forehead',
            'scalp')
        """
        from ..geometry import apply_vertex_mask
        out = apply_vertex_mask(name, v=self.v, tris=self.tris)
        self.v = out['v']
        self.tris = out['tris']

    @staticmethod
    @torch.inference_mode()
    def from_video(path, **kwargs):
        """Utility method to directly initialize a ``Data4D`` object by calling
        the ``videorecon`` function.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to video that will be reconstructed
        **kwargs
            Keyword arguments passed to ``videorecon``

        Returns
        -------
        data : Data4D
            A Data4D object
        """
        from ..recon import videorecon
        data = videorecon(path, **kwargs)
        return data

    @classmethod
    def load(cls, path, device=None):
        """Loads an HDF5 file from disk, parses its contents, and creates the
        initialization parameters necessary to initialize a ``*Data`` object.

        Parameters
        ----------
        path : str, pathlib.Path
            A path towards an HDF5 file data reconstructed by Medusa

        Returns
        -------
        An initialized Data4D object
        """

        init_kwargs = dict()
        with h5py.File(path, "r") as f_in:

            if device is None:
                device = f_in.attrs.get("device", DEVICE)

            for attr, data in f_in.items():
                if isinstance(data, h5py.Group):
                    data = dict(data.attrs)
                elif isinstance(data, h5py.Dataset):
                    data = torch.as_tensor(data[:], device=device)

                init_kwargs[attr] = data

            for attr, value in f_in.attrs.items():
                # Override device from file with provided parameter (if any)
                if attr == 'device':
                    value = device

                init_kwargs[attr] = value

        return cls(**init_kwargs)

    def to_local(self):
        """Converts the data to local space."""
        if self.space == 'local':
            LOGGER.warning("Data already in 'local' space!")
        else:
            self.v = transform_points(torch.inverse(self.mat), self.v)
            self.cam_mat = torch.linalg.inv(self.mat[0]) @ self.cam_mat
            self.cam_mat[3, :] = torch.tensor([0., 0., 0., 1.], device=self.device)
            self.space = "local"

    def to_world(self):
        """Converts the data to world space."""
        if self.space == 'world':
            LOGGER.warning("Data already in 'world' space!")
        else:
            self.v = transform_points(self.mat, self.v)
            self.cam_mat = self.mat[0] @ self.cam_mat
            self.cam_mat[3, :] = torch.tensor([0., 0., 0., 1.], device=self.device)
            self.space = "world"

    def project_to_68_landmarks(self):
        """Projects to 68 landmark set.

        Returns
        -------
        v_proj :
        """

        topo = self._infer_topo()
        if topo == 'mediapipe':
            fname = "mpipe/mediapipe_lmk68_embedding.npz"
        elif topo == 'flame-coarse':
            fname = "flame/flame_lmk68_embedding.npz"
        else:
            raise ValueError(f"No known embedding for {topo}")

        emb = np.load(Path(__file__).parents[1] / f"data/{fname}")
        face_idx = torch.as_tensor(
            emb['lmk_faces_idx'], dtype=torch.int64, device=self.device
        )

        # n_face x V x 3 (faces) x 3 (faces) x 3 (xyz)
        vf = self.v[:, self.tris[face_idx]]
        bcoords = torch.as_tensor(emb["lmk_bary_coords"], device=self.device)
        # n_face x 68 x 3
        v_proj = torch.sum(vf * bcoords[:, :, None], dim=2)

        return v_proj

    def get_face(self, index, pad_missing=True):
        """Get the data from a particular face in the reconstruction.

        Parameters
        ----------
        index : int
            Integer index corresponding to the face
        """
        available = self.face_idx.unique()
        if index not in available:
            raise ValueError(f"Face not available; choose from {available.tolist()}")

        f_idx = self.face_idx == index
        T = self.video_metadata['n_img']

        if pad_missing:
            shape = (T, *self.v.shape[1:])
            v = torch.full(shape, torch.nan, device=self.device)
            img_idx = self.img_idx[f_idx]
            v[img_idx] = self.v[f_idx]
            mat = torch.full((T, 4, 4), torch.nan, device=self.device)
            mat[img_idx] = self.mat[f_idx]
            img_idx = torch.arange(T, device=self.device)
            face_idx = torch.full((T,), index, device=self.device)
        else:
            v = self.v[f_idx]
            mat = self.mat[f_idx]
            img_idx = self.img_idx[f_idx]
            face_idx = self.face_idx[f_idx]

        init_kwargs = {
            'v': v,
            'mat': mat,
            'face_idx': face_idx,
            'img_idx': img_idx
        }
        init_kwargs = {**self.__dict__, **init_kwargs}
        return self.__class__(**init_kwargs)

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

        >>> from medusa.data import get_example_data4d
        >>> data = get_example_data4d(load=True, model="mediapipe")
        >>> params = data.decompose_mats(to_df=False)
        >>> params.shape
        (232, 12)
        """

        out = []  # maybe dict?
        for face_id in self.face_idx.unique():
            data = self.get_face(face_id)

            T = data.mat.shape[0]
            params = np.zeros((T, 12))
            for i in range(T):

                if torch.isnan(data.mat[i]).all():
                    params[i, :] = np.nan
                    continue

                mat = data.mat[i].cpu().numpy()
                scale, shear, angles, trans, _ = decompose_matrix(mat)
                params[i, :3] = trans
                params[i, 3:6] = np.rad2deg(angles)
                params[i, 6:9] = scale
                params[i, 9:12] = shear

            if to_df:
                cols = [
                    "Trans. X",
                    "Trans. Y",
                    "Trans. Z",
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

            out.append(params)

        if len(out) == 1:
            out = out[0]

        return out

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

        >>> from medusa.data import get_example_data4d
        >>> data = get_example_data4d(load=True, model="mediapipe")
        >>> orig_mats = data.mat.copy()
        >>> params = data.decompose_mats(to_df=False)
        >>> data.compose_mats(params)
        >>> np.testing.assert_array_almost_equal(orig_mats, data.mat)  # passes!
        """
        T = params.shape[0]
        mats = np.zeros((T, 4, 4))

        if isinstance(params, pd.DataFrame):
            params = params.to_numpy()

        for i in range(T):
            p = params[i, :]
            trans, rots, scale, shear = p[:3], p[3:6], p[6:9], p[9:]
            rots = np.deg2rad(rots)
            mats[i, :, :] = compose_matrix(scale, shear, rots, trans)

        self.mat = torch.as_tensor(mats, dtype=torch.float32, device=self.device)

    def filter_faces(self, present_threshold=0.1):
        """Filters the reconstructed faces by the proportion of frames they are
        present in.

        Parameters
        ----------
        present_threshold : float
            Lower bound on proportion present
        """
        keep = filter_faces(self.face_idx, self.video_metadata['n_img'], present_threshold)

        if not torch.all(keep):
            for attr in ('v', 'mat', 'img_idx', 'face_idx'):
                data = getattr(self, attr)
                if data.shape[0] == keep.shape[0]:
                    setattr(self, attr, data[keep])

        self.face_idx = _ensure_consecutive_face_idx(self.face_idx)

    def __getitem__(self, idx):

        kwargs = self.__dict__
        kwargs['v'] = self.v[idx]
        kwargs['mat'] = self.mat[idx]
        kwargs['img_idx'] = self.img_idx[idx]
        kwargs['face_idx'] = self.face_idx[idx]
        kwargs['video_metadata']['n_img'] = kwargs['img_idx'].max()

        return self.__class__(**kwargs)

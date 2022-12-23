import numpy as np
import torch
from trimesh import Trimesh

from ..data import get_flame_config, get_template_flame
from ..recon.flame.decoders import FLAME
from .fourD import Data4D
from ..data import get_template_mediapipe

from ..defaults import DEVICE

flame_path = get_flame_config("flame_path")
flame_generator = FLAME(flame_path, 300, 100)


class Base3D:
    def save(self, path, file_type="obj", **kwargs):
        mesh = Trimesh(self.v, self.f)
        with open(path, "w") as f_out:
            mesh.export(f_out, file_type=file_type, **kwargs)

    def animate(self, v, mat, is_deltas=True):

        if is_deltas:
            v = self.v + v

        for i in range(v.shape[0]):
            v_ = v[i, ...]
            v_ = np.c_[v_, np.ones(v_.shape[0])]
            v[i, ...] = (v_ @ mat[i, ...].T)[:, :3]

        return v


class Flame3D(Base3D):
    def __init__(self, v=None, mat=None, topo='coarse', device=DEVICE):

        data = get_template_flame(topo, keys=['v', 'tris'], device=device)
        self.v = data["v"] if v is None else v
        self.f = data["tris"]
        self.mat = torch.eye(4) if mat is None else mat

    @classmethod
    def from_4D(cls, data, index=0):
        v = data.v[index, ...]
        mat = data.mat[index, ...]
        return cls(v, mat)

    @classmethod
    def random(
        cls,
        shape=None,
        exp=None,
        pose=None,
        rot_x=None,
        rot_y=None,
        rot_z=None,
        no_exp=True,
    ):

        if shape is None:
            shape = torch.randn(1, 300) * 0.6

        if no_exp:
            exp = torch.zeros((1, 100))
        else:
            if exp is None:
                exp = torch.randn(1, 100) * 0.6

        for param in [shape, exp]:
            if param.ndim == 1:
                param = param.unsqueeze(0)

        if shape.shape[1] < 300:
            tmp = shape.clone()
            shape = torch.zeros((1, 300))
            shape[tmp.shape[1]] = tmp[0, :]

        if exp.shape[1] < 100:
            tmp = exp.clone()
            exp = torch.zeros((1, 100))
            exp[tmp.shape[1]] = tmp[0, :]

        if pose is None:
            pose = torch.zeros((1, 6))
            if rot_x is not None:
                pose[0, 0] = np.deg2rad(rot_x)

            if rot_y is not None:
                pose[0, 1] = np.deg2rad(rot_y)

            if rot_z is not None:
                pose[0, 2] = np.deg2rad(rot_z)
        else:
            if pose.ndim == 1:
                pose = pose.unsqueeze(0)

        v, mat = flame_generator(shape, exp, pose)
        v = v.squeeze().cpu().numpy()
        mat = mat.squeeze(0).mean(dim=0).cpu().numpy()  # 4 x 4
        return cls(v, mat, dense=False)

    def animate(self, v, mat, sf, frame_t, is_deltas=True):

        v = super().animate(v, mat, is_deltas)
        animated = Data4D(
            v=v, mat=mat, cam_mat=np.eye(4), space="world", sf=sf, frame_t=frame_t
        )
        return animated


class Mediapipe3D(Base3D):
    def __init__(self, v=None, mat=None):

        data = get_template_mediapipe()
        self.v = data["v"] if v is None else v
        self.f = data["f"]
        self.mat = np.eye(4) if mat is None else mat

    @classmethod
    def from_4D(cls, data, index=0):
        v = data.v[index, ...]
        mat = data.mat[index, ...]
        return cls(v, mat)

    def animate(self, v, mat, sf, frame_t, is_deltas=True):

        v = super().animate(v, mat, is_deltas)
        animated = Data4D(
            v=v, mat=mat, cam_mat=np.eye(4), space="world", sf=sf, frame_t=frame_t
        )
        return animated

import numpy as np
from trimesh import Trimesh

from .fourD import Flame4D, Mediapipe4D


class Base3D:

    def save(self, path, file_type='obj', **kwargs):
        mesh = Trimesh(self.v, self.f)        
        with open(path, 'w') as f_out:
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
    
    def __init__(self, v=None, mat=None, dense=False):
        from ..data import get_template_flame 
        data = get_template_flame(dense=dense)
        self.v = data['v'] if v is None else v
        self.f = data['f']
        self.mat = np.eye(4) if mat is None else mat

    @classmethod
    def from_4D(cls, data, index=0):
        v = data.v[index, ...]
        mat = data.mat[index, ...]
        return cls(v, mat)
    
    def animate(self, v, mat, sf, frame_t, is_deltas=True):
        
        v = super().animate(v, mat, is_deltas)
        animated = Flame4D(v=v, mat=mat, cam_mat=np.eye(4), space='world', sf=sf,
                               frame_t=frame_t)
        return animated


class Mediapipe3D(Base3D):
    
    def __init__(self, v=None, mat=None):
        from ..data import get_template_mediapipe
        data = get_template_mediapipe()
        self.v = data['v'] if v is None else v
        self.f = data['f']
        self.mat = np.eye(4) if mat is None else mat

    @classmethod
    def from_4D(cls, data, index=0):
        v = data.v[index, ...]
        mat = data.mat[index, ...]
        return cls(v, mat)        

    def animate(self, v, mat, sf, frame_t, is_deltas=True):
        
        v = super().animate(v, mat, is_deltas)
        animated = Mediapipe4D(v=v, mat=mat, cam_mat=np.eye(4), space='world', sf=sf,
                               frame_t=frame_t)
        return animated
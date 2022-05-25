import pickle
import warnings
import numpy as np
from pathlib import Path

from .data import get_template_flame, get_template_mediapipe
from .core4d import Flame4D, Mediapipe4D


class Base3D:

    def __init__(self):
        pass

    def save_obj(self):
        pass

    def render_image(self, f_out=None):
        pass        

    def animate(self):
        pass


class Flame3D(Base3D):
    
    def __init__(self, v=None, mat=None):
        
        data = get_template_flame()
        self.v = data['v'] if v is None else v
        self.mat = np.eye(4) if mat is None else mat

    @classmethod
    def from_4D(cls, data, index=0):
        v = data.v[index, ...]
        mat = data.mat[index, ...]
        return cls(v, mat)        


class Mediapipe3D(Base3D):
    
    def __init__(self, v=None, mat=None):
        
        data = get_template_mediapipe()
        self.v = data['v'] if v is None else v
        self.mat = np.eye(4) if mat is None else mat

    @classmethod
    def from_4D(cls, data, index=0):
        v = data.v[index, ...]
        mat = data.mat[index, ...]
        return cls(v, mat)        

    def animate(self, v, mat, sf, frame_t, is_deltas=True):
        
        if is_deltas:
            v = self.v + v
    
        for i in range(v.shape[0]):
            v_ = v[i, ...]
            v_ = np.c_[v_, np.ones(v_.shape[0])]
            v[i, ...] = (v_ @ mat[i, ...].T)[:, :3]

        animated = Mediapipe4D(v=v, mat=mat, cam_mat=np.eye(4), space='world', sf=sf,
                           frame_t=frame_t)
        return animated
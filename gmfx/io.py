import os
import h5py
import imageio
import numpy as np
import os.path as op
import pandas as pd
from tqdm import tqdm
from .constants import FACES

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Data:
    """ Generic Data class to store, load, and save vertex/face data. """
    def __init__(self, v=None, f=None, imgs=None, frame_t=None, events=None, fps=None,
                 dense=False, sub=None, run=None, task=None):

        self.v = v
        self.f = f
        self.imgs = imgs
        self.frame_t = frame_t
        self.events = events
        self.fps = fps
        self.dense = dense
        self.sub = sub
        self.run = run
        self.task = task
        self._check()

    def _check(self):

        self.v = self.v.astype(np.float32)

        if self.f is None:
            self.f = FACES['dense'] if self.dense else FACES['coarse']
            self.f = self.f.astype(np.int32)

        if self.frame_t is not None:
            pass
            #if self.frame_t.size != self.v.shape[0]:
            #    raise ValueError("Number of frame times does not equal "
            #                     "number of vertex time points!")
    
    @classmethod
    def load(cls, path):
        """ Loads a hdf5 file from disk and returns a Data object. """
        
        with h5py.File(path, "r") as f_in:
            v = f_in['v'][:]
            f = f_in['f'][:]
            
            frame_t = None
            if 'frame_t' in f_in:
                frame_t = f_in['frame_t'][:]

            dense = f_in['v'].attrs['dense']
            fps = f_in['frame_t'].attrs['fps']

        events = None
        if 'events' in f_in:
            events = pd.read_hdf(path, key='/events')

        data = cls(v, f, frame_t, events, fps, dense)
        return data
        
    def save(self, path):
        """ Saves data to disk as a hdf5 file. """

        out_dir = op.dirname(path)
        if not op.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with h5py.File(path, 'w') as f_out:
            for attr in ['v', 'f', 'frame_t']:
                data = getattr(self, attr)
                if data is not None:
                    f_out.create_dataset(attr, data=data)

            f_out['v'].attrs['dense'] = self.dense
            f_out['frame_t'].attrs['fps'] = self.fps

        # Note to self: need to do this outside h5py.File context,
        # because Pandas assumes a buffer or path
        if self.events is not None:
            self.events.to_hdf(path, key='/events', mode='a')

    def visualize(self, path):
        """ Plots each time points separately and creates a movie. """
        
        writer = imageio.get_writer(path, mode='I', fps=self.fps)
        for i in tqdm(range(self.v.shape[0])):
            v = self.v[i, ...]
            res = render(self.imgs[i, ...], [v.T], self.f.copy(order='C'), with_bg_flag=True)
            writer.append_data(res)

        writer.close()
        
    def __len__(self):
        return self.v.shape[0]
    
    def __getitem__(self, idx):
        return self.v[idx, :, :]
    
    def __setitem__(self, idx, v):
        self.v[idx, ...] = v
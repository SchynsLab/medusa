import os
import torch
import h5py
import numpy as np
import os.path as op
import pandas as pd

from .constants import FACES


class Data:
    """ Generic Data class to store, load, and save vertex/face data. 
    
    Parameters
    ----------
    v : ndarray
        Numpy array of shape T (time points) x nV (no. vertices) x 3 (x/y/z)
    motion : ndarray
        Numpy array of shape T (time points) x 6 (global rot x/y/z, trans x/y, scale z)
    f : ndarray
        Numpy array of shape nF (no. faces) x 3 (vertices per face)
    frame_t : ndarray
        Numpy array of length T (time points) with "frame times", i.e.,
        onset of each frame (in seconds) from the video
    """
    def __init__(self, v=None, motion=None, f=None, frame_t=None, events=None,
                 fps=None, dense=False, sub=None, run=None, task=None):

        self.v = v
        self.motion = motion
        self.f = f
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
            motion = f_in['motion'][:]
            f = f_in['f'][:]
            
            frame_t = None
            if 'frame_t' in f_in:
                frame_t = f_in['frame_t'][:]

            dense = f_in['v'].attrs['dense']
            fps = f_in['frame_t'].attrs['fps']

        events = None
        if 'events' in f_in:
            events = pd.read_hdf(path, key='/events')

        data = cls(v, motion, f, frame_t, events, fps, dense)
        return data
        
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

            f_out['v'].attrs['dense'] = self.dense
            f_out['frame_t'].attrs['fps'] = self.fps

        # Note to self: need to do this outside h5py.File context,
        # because Pandas assumes a buffer or path
        if self.events is not None:
            self.events.to_hdf(path, key='/events', mode='a')

    def __len__(self):
        return self.v.shape[0]
    
    def __getitem__(self, idx):
        return self.v[idx, :, :]
    
    def __setitem__(self, idx, v):
        self.v[idx, ...] = v
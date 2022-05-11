import mne
import numpy as np


class EpochsArray(mne.epochs.EpochsArray):
    """ Custom EpochsArray, with """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @classmethod
    def from_gmfx(cls, v, sf, tmin=-.5, includes_motion=False):
        
        v = v.copy()
        # N (trails), T (time points), nV (number of vertices)
        N, T, nV = v.shape[:3]

        # Flatten vertices and coord (XYZ) dimensions
        v = v.reshape((N, T, -1))

        # N x T x (V x 3) --> N x (V x 3) x T
        # (as is expected by MNE)
        v = np.transpose(v, (0, 2, 1))
        
        if includes_motion:
            ch_names = [f'v{i}_{c}' for i in range((nV - 12) // 3) for c in ['x', 'y', 'z']]
            ch_names += ['xt', 'yt', 'zt', 'xr', 'yr', 'zr', 'xs', 'ys', 'zs', 'xsh', 'ysh', 'zsh']
        else:
            ch_names = [f'v{i}_{c}' for i in range(nV // 3) for c in ['x', 'y', 'z']]
                
        info = mne.create_info(
            # vertex 0 (x), vertex 0 (y), vertex 0 (z), vertex 1 (x), etc
            ch_names=ch_names,
            ch_types=['misc'] * v.shape[1],
            sfreq=sf
        )        
        
        return cls(v, info, tmin=tmin, verbose='WARNING')
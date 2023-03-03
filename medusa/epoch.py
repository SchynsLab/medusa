import numpy as np
import pandas as pd
from pathlib import Path
from medusa.containers import Data4D
from scipy.interpolate import interp1d

from .transforms import compose_matrix


class EpochsArray:

    def __init__(self, v_epochs, params_epochs, frame_t):
        self.v_epochs = v_epochs
        self.params_epochs = params_epochs
        self.frame_t = frame_t

    def baseline_normalize(self):
        pass

    def to_mne(self):
        pass

    @classmethod
    def from_4D(cls, data, events, start=-.5, end=5., period=0.01, T=50, anchor='onset'):

        # Read in pandas DataFrame if `events` is a string/Path
        if isinstance(events, (str, Path)):
            events = pd.read_csv(events, sep='\t')

        # How many events do we have?
        N = events.shape[0]

        if not isinstance(data, list):
            # Single Data4D object
            data = [data]
            events = [events]
        else:
            # Multiple Data4D objects with one event file
            events = [events.iloc[[i], :] for i in range(events.shape[0])]

        if anchor != 'duration':
            T = int(round((end - start) / period + 1))

        v_epochs = []
        params_epochs = []
        for d, ev in zip(data, events):

            if isinstance(d, (Path, str)):
                d = Data4D.load(d)

            if d.space != 'local':
                d.to_local()

            assert(d.face_idx.unique().numel() == 1)

            fps = d.video_metadata['fps']
            n_frames = d.video_metadata['n_img']
            frame_t = np.linspace(0, n_frames / fps, n_frames, endpoint=False)
            v = d.v.cpu().numpy()
            params = d.decompose_mats(to_df=False)

            # Define interpolator; for now, default is linear
            ipl_v = interp1d(frame_t, v, axis=0, kind="linear", bounds_error=False)
            ipl_p = interp1d(frame_t, params, axis=0, kind="linear", bounds_error=False)

            for ii in range(ev.shape[0]):

                row = ev.loc[ev.index[ii], ['onset', 'duration']]
                if isinstance(row['onset'], (int, np.int64)):
                    row /= fps

                if anchor == 'onset':
                    anchor_ =  row['onset']
                    range_ = (anchor_ + start, anchor_ + end)
                elif anchor == 'offset':
                    anchor_ = row['onset'] + row['duration']
                    range_ = (anchor_ + start, anchor_ + end)
                elif anchor == 'duration':
                    onset = row['onset']
                    range_ = (onset, onset + row['duration'])
                else:
                    raise ValueError(f"Unknown anchor type: {anchor}, choose from "
                                      "('onset', 'offset', 'duration')")

                event_t = np.linspace(*range_, endpoint=True, num=T)
                v_epoch = ipl_v(event_t)
                params_epoch = ipl_p(event_t)

                v_epochs.append(v_epoch)
                params_epochs.append(params_epoch)

        v_epochs = np.stack(v_epochs, axis=0)
        params_epochs = np.stack(params_epochs, axis=0)

        return cls(v_epochs, params_epochs, frame_t)

    def to_4D(self, agg='mean', device='cuda'):

        if agg == 'mean':
            agg_f = np.nanmean
        elif agg == 'median':
            agg_f = np.nanmedian
        else:
            raise ValueError(f"Unknown aggregation method: {agg}, choose from ('mean', 'median')")

        v_av = agg_f(self.v_epochs, axis=0)
        params_av = agg_f(self.params_epochs, axis=0)
        params_av[:] = params_av[None, 0, :]
        p = params_av[0]
        trans, rots, scale, shear = p[:3], p[3:6], p[6:9], p[9:]
        rots = np.deg2rad(rots)
        mat0 = compose_matrix(scale, shear, rots, trans)
        cam_mat = np.eye(4)
        cam_mat[2, 3] = 4
        cam_mat = np.linalg.inv(mat0) @ cam_mat
        cam_mat[3, :] = [0., 0., 0., 1.]
        data = Data4D(v_av, params_av, cam_mat=cam_mat, space='local', device=device)

        return data

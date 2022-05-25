import h5py
import numpy as np
from pathlib import Path  


class EpochsArray:
    """Custom EpochsArray, with some extra functionality to interact with
    medusa.

    Parameters
    ----------
    v : np.ndarray
        A 4D numpy array of shape N (events/trails) x T (time points)
        x nV (number of vertices) x 3 (X, Y, Z)
    sf : float
        Sampling frequency of the data (`v`)
    events : pd.DataFrame
        events : pd.DataFrame
        A BIDS-style DataFrame with event (trial) information,
        with at least the columns 'onset' and 'trial_type'
    frame_t : np.ndarray
        A 1D numpy array with the onset of each frame from
        the video that was reconstructed
    """

    def __init__(self, v, params, frame_t, recon_model_name, events=None):        
        self.v = v
        self.params = params
        self.frame_t = frame_t
        self.recon_model_name = recon_model_name
        self.sf = np.diff(frame_t).mean()
        self.events = events
    
    def save(self, path, compression_level=9):
        """ Saves (meta)data to disk as an HDF5 file.

        Parameters
        ----------
        path : str
            Path to save the data to
        compression_level : int
            Level of compression (higher = more compression, but slower; max = 9)
        """

        out_dir = Path(path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f_out:

            for attr in ["v", "params", "frame_t"]:
                data = getattr(self, attr, None)
                data = data.astype(np.float32)
                f_out.create_dataset(attr, data=data, compression=compression_level)

            for attr in ["sf", "recon_model_name"]:
                f_out.attrs[attr] = getattr(self, attr)

            f_out.attrs["path"] = path

        # Note to self: need to do this outside h5py.File context,
        # because Pandas assumes a buffer or path, not an
        if self.events is not None:
            self.events.to_hdf(path, key="events", mode="a")

    def to_mne(self, frame_t, include_global_motion=True):
        """ Initalize a MNE EpochsArray.

        Parameters
        ----------
        include_global_motion : bool
            Whether to add global motion ('mat') to the data as if it were a separate
            set of channels  

        Returns
        -------
        An instance of the EpochsArray class
        """
        
        try:
            import mne
        except ImportError:
            raise ValueError("MNE is not installed!")    
            
        v = self.v.copy()

        # N (trails), T (time points), nV (number of vertices)
        N, T, nV = v.shape[:3]

        # Flatten vertices and coord (XYZ) dimensions
        v = v.reshape((N, T, -1))
        if include_global_motion:
            v = np.c_[v, self.params]
            nV = nV + 12
        
        # N x T x (V x 3) --> N x (V x 3) x T
        # (as is expected by MNE)
        v = np.transpose(v, (0, 2, 1))

        if include_global_motion:
            ch_names = [
                f"v{i}_{c}" for i in range(nV - 12) for c in ["x", "y", "z"]
            ]
            ch_names += [
                "xt",
                "yt",
                "zt",
                "xr",
                "yr",
                "zr",
                "xs",
                "ys",
                "zs",
                "xsh",
                "ysh",
                "zsh",
            ]
        else:
            ch_names = [f"v{i}_{c}" for i in range(nV) for c in ["x", "y", "z"]]
        
        info = mne.create_info(
            # vertex 0 (x), vertex 0 (y), vertex 0 (z), vertex 1 (x), etc
            ch_names=ch_names,
            ch_types=["misc"] * v.shape[1],
            sfreq=self.sf,
        )

        if self.events is not None:
            events_, event_id = self._events_to_mne(frame_t)
        else:
            events_, event_id = None, None
        
        tmin = self.frame_t.min()
        return mne.epochs.EpochsArray(
            v, info, tmin=tmin, events=events_, event_id=event_id,
            verbose="WARNING"
        )

    def _events_to_mne(self, frame_t):
        """Converts events DataFrame to (N x 3) array that
        MNE expects.

        Returns
        -------
        events_ : np.ndarray
            An N (number of trials) x 3 array, with the first column
            indicating the sample *number* (not time) and the third
            column indicating the sample condition (see the returned
            `event_id` dictionary for the mapping between condition number
            and string representation)
        event_id : dict
            A dictionary with condition strings as keys and condition numbers
            as values; the values correspond to the third column of `events_`
        """

        events = self.events

        event_id = {k: i for i, k in enumerate(events["trial_type"].unique())}
        events_ = np.zeros((events.shape[0], 3))
        for i, (_, ev) in enumerate(events.iterrows()):
            events_[i, 2] = event_id[ev["trial_type"]]
            t_diff = np.abs(frame_t - ev["onset"])
            events_[i, 0] = np.argmin(t_diff)

            if np.min(t_diff) > 0.05:
                min_ = np.min(t_diff).round(4)
                raise ValueError(
                    f"Nearest sample is {min_} seconds away "
                    f"for trial {i+1}; try resampling the data to a "
                    "higher resolution!"
                )

        events_ = events_.astype(np.int64)

        return events_, event_id
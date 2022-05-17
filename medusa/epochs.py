import mne
import numpy as np


class EpochsArray(mne.epochs.EpochsArray):
    """Custom EpochsArray, with some extra functionality to interact with
    medusa.

    Parameters
    ----------
    args : list
        Positional parameters to be passed to initialization of the
        MNE EPochsArray (the base class)
    kwargs : list
        Keyword parameters to be passed to initialization of the
        MNE EPochsArray (the base class)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_medusa(
        cls, v, sf, events=None, frame_t=None, tmin=-0.5, includes_motion=False
    ):
        """Classmethod to initalize an EpochsArray from medusa data.

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
        tmin : float
            Start (in seconds) of each epoch relative to stimulus onset
        includes_motion : bool
            Whether the data (`v`) also includes the epoched motion parameters;
            if so, it is assumed that the last 12 values in the third dimension
            of `v` represents the motion parameters

        Returns
        -------
        An instance of the EpochsArray class
        """
        v = v.copy()

        # N (trails), T (time points), nV (number of vertices)
        N, T, nV = v.shape[:3]

        # Flatten vertices and coord (XYZ) dimensions
        v = v.reshape((N, T, -1))

        # N x T x (V x 3) --> N x (V x 3) x T
        # (as is expected by MNE)
        v = np.transpose(v, (0, 2, 1))

        if includes_motion:
            ch_names = [
                f"v{i}_{c}" for i in range((nV - 12) // 3) for c in ["x", "y", "z"]
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
            ch_names = [f"v{i}_{c}" for i in range(nV // 3) for c in ["x", "y", "z"]]

        info = mne.create_info(
            # vertex 0 (x), vertex 0 (y), vertex 0 (z), vertex 1 (x), etc
            ch_names=ch_names,
            ch_types=["misc"] * v.shape[1],
            sfreq=sf,
        )

        if events is not None:
            events_, event_id = cls.events_to_mne(events, frame_t)
        else:
            events_, event_id = None, None

        return cls(
            v, info, tmin=tmin, events=events_, event_id=event_id, verbose="WARNING"
        )

    @staticmethod
    def events_to_mne(events, frame_t):
        """Converts events DataFrame to (N x 3) array that
        MNE expects.

        Parameters
        ----------
        events : pd.DataFrame
            A BIDS-style DataFrame with event (trial) information,
            with at least the columns 'onset' and 'trial_type'
        frame_t : np.ndarray
            A 1D numpy array with the onset of each frame from
            the video that was reconstructed; necessary for
            converting event onsets in seconds to event onsets
            in samples (TODO: use sf for this?)

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

from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from ..epochs import EpochsArray
from ..containers import Data4D


def epoch(
    data,
    start=-0.5,
    end=3.0,
    period=0.01,
    baseline_correct=False,
    baseline_window=(None, None),
    baseline_mode="mean",
    add_back_grand_mean=False,
):
    """Creates epochs of the data.

    Parameters
    ----------
    data : str, Data
        Either a path (``str`` or ``pathlib.Path``) to a ``medusa`` hdf5
        data file or a ``Data`` object (like ``FlameData`` or ``MediapipeData``)
    start : float
        Start of the epoch (in seconds) relative to stimulus onset
    end : float
        End of the epoch (in seconds) relative to stimulus onset
    baseline_correct : bool
        Whether to apply baseline correction
    baseline_window : tuple[float]
        Tuple with two values, indicating baseline start and end (in seconds),
        respectively; if the first value is None, then the start is the beginning
        of the epoch; if the second value is None, then the end is at stimulus onset
        (i.e., 0)
    baseline_mode : str
        How to perform baseline correction (options: 'mean', 'ratio')
    add_back_grand_mean : bool
        Whether to add back the grand mean (average across all events and across
        the entire time series); if ``False``, the baseline of each event is centered
        around zero; if ``True``, the baseline of each event is centered around the grand
        mean of all events

    Returns
    -------
    epochsarray : medusa.epochs.EpochsArray
        An EpochsArray object
    """

    if isinstance(data, (str, Path)):
        # if data is a path to a hdf5 file, load it
        # (used by CLI)
        data = Data4D.load(data)

    if data.events is None:
        raise ValueError("Cannot epoch data without events!")

    if not data.events.shape[0]:
        raise ValueError("No events in data.events!")

    # D = data, flattened across vertices and coords
    D = data.v.reshape((data.v.shape[0], -1))

    # Add motion parameters to data
    motion = data.mats2params()
    D = np.c_[D, motion]

    # Define interpolator; for now, default is linear
    interpolator = interp1d(data.frame_t, D, axis=0, kind="linear")

    # epochs: N (stimuli) x T (time points) x V (vertices) x 3 (XYZ)
    N = data.events.shape[0]
    T = int(round((end - start) / period + 1))
    epochs = np.zeros((N, T, *D.shape[1:]))

    # Loop over trials (onsets)

    desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ]  Epoch trials")
    for i, onset in tqdm(data.events["onset"].iteritems(), total=N, desc=desc):

        # Define this trial's epoch window and times
        this_start = onset + start
        this_end = onset + end
        ft = np.linspace(this_start, this_end, endpoint=True, num=T)
        ft = ft[ft < data.frame_t[-1]]  # trim off frame times beyond acq

        # Perform actual epoching by interpolation
        epoched = interpolator(ft)

        if epoched.shape[0] < T:
            # If time points were trimmed off, add these back as NaNs
            missing = np.full([T - epoched.shape[0], *epoched.shape[1:]], np.nan)
            # missing = np.repeat(epoched[-1, None, :, :], T - epoched.shape[0], axis=0)
            epoched = np.r_[epoched, missing]

        # Store in 4D (N, T, V, 3) epochs array
        epochs[i, ...] = epoched

    if baseline_correct:
        # Implementation based on MNE, with some tweaks
        b_start, b_end = baseline_window
        if b_start is None:
            # baseline start at epoch start
            b_start = start

        if b_end is None:
            # baseline end at stimulus onset
            b_end = 0.0

        # extract baseline based on window
        t = np.linspace(start, end, endpoint=True, num=T)
        baseline = epochs[:, (t >= b_start) & (t <= b_end), ...]

        # Note that we baseline-correct each epoch, but then
        # add back the "grand mean" (across epochs) so that
        # our data is still in interpretable (and renderable) units
        if baseline_mode == "mean":
            epochs -= baseline.mean(axis=1, keepdims=True)
            if add_back_grand_mean:
                epochs += baseline.mean(axis=(0, 1))  # 'grand mean'
        elif baseline_mode == "ratio":
            epochs /= baseline.mean(axis=1, keepdims=True)
            if add_back_grand_mean:
                epochs *= baseline.mean(axis=(0, 1))
        else:
            raise NotImplementedError

    frame_t = np.linspace(start, end, endpoint=True, num=T)
    v = epochs[..., :-12].reshape((N, T, data.v.shape[1], 3))
    params = epochs[..., -12:]
    epochs_arr = EpochsArray(
        v=v,
        params=params,
        frame_t=frame_t,
        events=data.events,
        recon_model=data.recon_model,
    )

    data.v = v.mean(axis=0)
    data.render_video("test_av.gif", scaling=0.5, smooth=False, overlay=None)

    return epochs_arr

    # mne_epochs_arr = epochs_arr.to_mne(data.frame_t)
    # evoked = mne_epochs_arr.average(picks='misc')
    # fig = evoked.plot(picks='misc')
    # fig.savefig('test.png')
    # av = np.nanmean(epochs, axis=0)
    # data.v = av[:, :-12].reshape((av.shape[0], -1, 3))
    # data.params2mats(av[:, -12:])

    # from ..core3d import Mediapipe3D
    # neutral = Mediapipe3D()
    # data = neutral.animate(v=av[:, :-12].reshape((av.shape[0], -1, 3)),
    #                         mat=data.mat, sf=data.sf, is_deltas=True, frame_t=frame_t)
    # data.render_video('test.gif')
    # for i in range(data.v.shape[0]):
    #    v_ = data.v[i, ...]
    #    v_ = np.c_[v_, np.ones(v_.shape[0])]
    #    data.v[i, ...] = (v_ @ data.mat[i, ...].T)[:, :3]

    # data.cam_mat = data.mat[0, ...] @ data.cam_mat
    # data.space = 'world'
    # data.sf = 1 / period
    # data.render_video('test.gif', video=None)

    # epochsarray = EpochsArray()
    # Create (custom) EpochsArray and save
    # epochs_array = EpochsArray.from_medusa(
    #     epochs,
    #     events=data.events,
    #     frame_t=data.frame_t,
    #     sf=1 / period,
    #     tmin=start,
    #     includes_motion=True,
    # )
    # epochs_array.save(
    #     data.path.replace("_shape.h5", "_epo.fif"),
    #     split_size="2GB",
    #     fmt="single",
    #     overwrite=True,
    #     split_naming="bids",
    #     verbose="WARNING",
    # )

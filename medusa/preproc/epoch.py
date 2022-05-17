import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import interp1d

from ..utils import get_logger
from ..core import load_h5
from ..epochs import EpochsArray


def epoch(
    data,
    start=-0.5,
    end=3.0,
    period=0.01,
    align_peaks=False,
    max_shift=0.5,
    baseline_correct=False,
    baseline_window=(None, None),
    baseline_mode="mean",
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
    align_peaks : bool
        Whether to align peaks across epochs (not implemented yet)
    max_shift : float
        Maximum allowed shift (in seconds) for peak alignment; if `peak_align`
        is False, then this argument is ignored
    baseline_correct : bool
        Whether to apply baseline correction
    baseline_window : tuple[float]
        Tuple with two values, indicating baseline start and end (in seconds),
        respectively; if the first value is None, then the start is the beginning
        of the epoch; if the second value is None, then the end is at stimulus onset
        (i.e., 0)
    baseline_mode : str
        How to perform baseline correction (options: 'mean', 'ratio')
    """

    logger = get_logger()

    # TODO: rename .mat to .affine (cf. nifti)

    if isinstance(data, (str, Path)):
        # if data is a path to a hdf5 file, load it
        # (used by CLI)
        logger.info(f"Loading data from {data} ...")
        data = load_h5(data)

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
    T = int((end - start) / period + 1)
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

    if align_peaks:
        # TODO in the future (maybe)
        pass

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
            epochs += baseline.mean(axis=(0, 1))  # 'grand mean'
        elif baseline_mode == "ratio":
            epochs /= baseline.mean(axis=1, keepdims=True)
            epochs *= baseline.mean(axis=(0, 1))
        else:
            raise NotImplementedError

    # av = np.nanmean(epochs, axis=0)
    # data.v = av[:, :-12].reshape((av.shape[0], -1, 3))
    # data.params2mats(av[:, -12:])

    # for i in range(data.v.shape[0]):
    #    v_ = data.v[i, ...]
    #    v_ = np.c_[v_, np.ones(v_.shape[0])]
    #    data.v[i, ...] = (v_ @ data.mat[i, ...].T)[:, :3]

    # data.sf = 1 / period
    # data.render_video('test.gif', video=None)

    # Create (custom) EpochsArray and save
    epochs_array = EpochsArray.from_medusa(
        epochs,
        events=data.events,
        frame_t=data.frame_t,
        sf=1 / period,
        tmin=start,
        includes_motion=True,
    )
    epochs_array.save(
        data.path.replace("_shape.h5", "_epo.fif"),
        split_size="2GB",
        fmt="single",
        overwrite=True,
        split_naming="bids",
        verbose="WARNING",
    )

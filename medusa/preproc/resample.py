from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from ..io import load_h5


def resample(data, sampling_freq=None, kind="pchip"):
    """Resamples the data to a given sampling rate. This function can be used
    to resample the time points to a higher temporal resolution and/or a
    constant sampling period, which may not be the case for data that is
    acquired using a webcam.

    Parameters
    ----------
    data : str, Data
        Either a path (``str`` or ``pathlib.Path``) to a ``medusa`` hdf5
        data file or a ``Data`` object (like ``FlameData`` or ``MediapipeData``)
    sampling_freq : int
        Desired sampling frequency (in Hz); if `None` (default), the inverse
        of the (average) sampling period will be used
    kind : str
        Kind of interpolation to use, either 'pchip' (default), 'linear', 'quadratic',
        or 'cubic'

    Returns
    -------
    data : medusa.core.*Data
        An object with a class inherited from ``medusa.core.BaseData``

    Examples
    --------
    Resample a time series of 3D meshes to a sampling frequency of 50 Hz

    >>> from medusa.data import get_example_h5
    >>> data = get_example_h5(load=True, model='mediapipe')
    >>> np.round(data.sf, 2)  # sampling frequency
    23.98
    >>> data.v.shape[0]  # how many time points?
    232
    >>> data = resample(data, sampling_freq=50)
    >>> data.sf  # corresponds to desired sampling_freq!
    50
    >>> data.v.shape[0]  # how many time points after resampling?
    482
    """

    if isinstance(data, (str, Path)):
        # if data is a path to a hdf5 file, load it
        # (used by CLI)
        data = load_h5(data)

    ft = data.frame_t

    if sampling_freq is None:
        # np.diff gives sampling *period*
        sampling_period = np.mean(np.diff(ft))
        sampling_freq = 1 / sampling_period
        data.logger.info(f"Using the average sampling rate ({sampling_freq:.2f} Hz)")
    else:
        sampling_period = 1 / sampling_freq

    # Update
    data.sf = sampling_freq

    # Create interpolator with current ft and data
    if kind == "pchip":
        interpolator = PchipInterpolator(ft, data.v)
    else:
        interpolator = interp1d(ft, data.v, axis=0, kind=kind)

    # Determine new number of time points (`new_T`) and corresponding
    # new frame times (`new_ft`), which are all spaced
    # `sampling_period` apart
    new_T = int(np.ceil((ft[-1] - ft[0]) / sampling_period))
    new_ft = np.linspace(
        ft[0], ft[0] + new_T * sampling_period, endpoint=False, num=new_T
    )

    # Interpolate with new frametimes
    # Note: need to cast to float32 otherwise renderer crashes
    data.v = interpolator(new_ft).astype(np.float32)

    # Also interpolate motion
    motion = data.decompose_mats()

    if kind == "pchip":
        interpolator = PchipInterpolator(ft, motion)
    else:
        interpolator = interp1d(ft, motion, axis=0, kind=kind)

    motion = interpolator(new_ft).astype(np.float32)
    data.compose_mats(motion)  # updates .mat

    data.frame_t = new_ft  # save new frame times!

    return data

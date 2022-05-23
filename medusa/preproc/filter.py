import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

from ..core import load_h5
from ..utils import get_logger


def filter(data, low_pass, high_pass):
    """Applies a bandpass filter the vertex coordinate time series.
    Implementation based on https://stackoverflow.com/questions/
    12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

    Parameters
    ----------
    data : str, Data
        Either a path (``str`` or ``pathlib.Path``) to a ``medusa`` hdf5
        data file or a ``Data`` object (like ``FlameData`` or ``MediapipeData``)
    low_pass : float
        Low-pass cutoff (in Hz)
    high_pass : float
        High-pass cutoff (in Hz)

    Returns
    -------
    data : medusa.core.*Data
        An object with a class inherited from ``medusa.core.BaseData``
    """

    logger = get_logger()

    if isinstance(data, (str, Path)):
        # if data is a path to a hdf5 file, load it
        # (used by CLI)
        logger.info(f"Loading data from {data} ...")
        data = load_h5(data)

    nyq = 0.5 * data.sf  # sampling freq
    low = high_pass / nyq
    high = low_pass / nyq
    sos = butter(5, [low, high], analog=False, btype="band", output="sos")

    to_filter = [data.v, data.mats2params().to_numpy()]
    for i in range(len(to_filter)):
        d = to_filter[i]
        d_ = d.reshape((d.shape[0], -1))
        mu = d_.mean(axis=0)
        d_ = d_ - mu

        d_ = sosfilt(sos, d_, axis=0)

        # Undo normalization to get data back on original scale
        to_filter[i] = (d_ + mu).reshape(d.shape).astype(np.float32)

    data.v = to_filter[0]
    data.params2mats(to_filter[1])

    return data
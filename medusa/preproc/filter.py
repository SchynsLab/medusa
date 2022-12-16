from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from ..containers import Data4D


def bw_filter(data, low_pass, high_pass):
    """Applies a bandpass filter the vertex coordinate time series.
    Implementation based on `this StackOverflow post`

    <https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-
    butterworth-filter-with-scipy-signal-butter>`_.

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

    Examples
    --------
    Filter the data wit a high-pass of 0.005 Hz and a low-pass of 4 Hz:

    >>> from medusa.data import get_example_h5
    >>> data = get_example_h5(load=True, model='mediapipe')
    >>> data = bw_filter(data, low_pass=4., high_pass=0.005)
    """

    if isinstance(data, (str, Path)):
        # if data is a path to a hdf5 file, load it
        # (used by CLI)
        data = Data4D(data)

    nyq = 0.5 * data.sf  # sampling freq
    low = high_pass / nyq
    high = low_pass / nyq
    sos = butter(5, [low, high], analog=False, btype="band", output="sos")

    to_filter = [data.v, data.decompose_mats(to_df=False)]
    for i in range(len(to_filter)):
        d = to_filter[i]
        d_ = d.reshape((d.shape[0], -1))
        mu = d_.mean(axis=0)
        d_ = d_ - mu
        d_ = sosfilt(sos, d_, axis=0)

        # Undo normalization to get data back on original scale
        to_filter[i] = (d_ + mu).reshape(d.shape).astype(np.float32)

    data.v = to_filter[0]
    data.compose_mats(to_filter[1])

    return data


class OneEuroFilter:
    """Based on https://github.com/jaantollander/OneEuroFilter."""
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        # Previous values.
        self.x_prev = x0.astype(np.float64)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    @staticmethod
    def smoothing_factor(t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    @staticmethod
    def exponential_smoothing(a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        if np.any(np.isnan(dx)):
            dx = np.zeros_like(dx)
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

import torch
import numpy as np
from scipy.signal import butter, sosfilt


def bw_filter(data, fps, low_pass, high_pass):
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

    convert_back_to_tensor = False
    if torch.is_tensor(data):
        convert_back_to_tensor = True
        device = data.device
        dtype = data.dtype
        data = data.cpu().numpy()

    nyq = 0.5 * fps  # sampling freq
    low = high_pass / nyq
    high = low_pass / nyq
    sos = butter(5, [low, high], analog=False, btype="band", output="sos")

    mu = data.mean(axis=0, keepdims=True)
    data = data - mu
    data = sosfilt(sos, data, axis=0)

    # Undo normalization to get data back on original scale
    data = data + mu

    if convert_back_to_tensor:
        data = torch.as_tensor(data, dtype=dtype, device=device)

    return data


class OneEuroFilter:
    """A high-pass filter that can be used in real-time applications; based on
    the implementation by `Jaan Tollander.

    <https://github.com/jaantollander/OneEuroFilter>`_.

    Parameters
    ----------
    TODO
    """

    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        # Previous values
        self.x_prev = x0.astype(np.float64)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    @staticmethod
    def smoothing_factor(t_e, cutoff):
        """Apply smoothing factor."""
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    @staticmethod
    def exponential_smoothing(a, x, x_prev):
        """Apply exponential smoothing."""
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

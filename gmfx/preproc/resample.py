import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d, PchipInterpolator

from ..io import load_h5
from ..utils import get_logger

logger = get_logger()


def resample(data, sampling_freq=None, kind='pchip', video=None):
    """ Resamples the data to a given sampling rate.
    This function can be used to resample the time points
    to a higher temporal resolution and/or a constant
    sampling period, which may not be the case for data
    that is acquired using a webcam.
    
    Parameters
    ----------
    data : str, Data
        Either a path (str, pathlib.Path) to a `gmfx` hdf5 data file
        or a gmfx.io.Data object (i.e., data loaded from the hdf5 file)
    sampling_freq : int
        Desired sampling frequency (in Hz); if `None` (default), the inverse
        of the (average) sampling period will be used
    kind : str
        Kind of interpolation to use, either 'pchip' (default), 'linear', 'quadratic',
        or 'cubic'
    """

    if isinstance(data, (str, Path)):
        # if data is a path to a hdf5 file, load it
        # (used by CLI)
        logger.info(f"Loading data from {data} ...")
        data = load_h5(data)
    
    ft = data.frame_t
    
    if sampling_freq is None:
        # np.diff gives sampling *period*
        sampling_period = np.mean(np.diff(ft))
        sampling_freq = 1 / sampling_period
        logger.info(f"Using the average sampling rate ({sampling_freq:.2f} Hz)")
    else:
        sampling_period = 1 / sampling_freq

    # Update
    data.sf = sampling_freq

    # Create interpolator with current ft and data
    if kind == 'pchip':
        interpolator = PchipInterpolator(ft, data.v)
    else:
        interpolator = interp1d(ft, data.v, axis=0, kind=kind)

    # Determine new number of time points (`new_T`) and corresponding
    # new frame times (`new_ft`), which are all spaced
    # `sampling_period` apart
    new_T = int(np.ceil((ft[-1] - ft[0]) / sampling_period))
    new_ft = np.linspace(ft[0], ft[0] + new_T * sampling_period, endpoint=False, num=new_T)
    
    # Interpolate with new frametimes
    # Note: need to cast to float32 otherwise renderer crashes
    data.v = interpolator(new_ft).astype(np.float32)
    
    # Also interpolate motion   
    if kind == 'pchip':
        interpolator = PchipInterpolator(ft, data.motion)
    else:
        interpolator = interp1d(ft, data.motion, axis=0, kind=kind)
    
    data.motion = interpolator(new_ft).astype(np.float32)
    data.frame_t = new_ft

    if data.path is None:
        logger.warning("Input `data` has no known save path (`path` attribute); "
                       "saving to current directory!")
    
    # Save!
    pth = data.path
    desc = 'desc-' + pth.split('desc-')[1].split('_')[0] + '+interp'
    f_out = pth.split('desc-')[0] + desc
    data.plot_data(f_out + '_qc.png', plot_motion=True, plot_pca=True, n_pca=3)
    data.render_video(f_out + '_shape.gif', video=video)
    data.save(f_out + '_shape.h5')
    
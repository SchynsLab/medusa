import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

from ..io import Data
from ..utils import get_logger

logger = get_logger()

def filter(data, low_pass, high_pass, device='cuda'):
    """ Applies a bandpass filter the vertex coordinate time series.
    Implementation based on https://stackoverflow.com/questions/
    12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    
    Parameters
    ----------
    data : str, Data
        Either a path (str, pathlib.Path) to a `gmfx` hdf5 data file
        or a gmfx.io.Data object (i.e., data loaded from the hdf5 file)
    low_pass : float
        Low-pass cutoff (in Hz)
    high_pass : float
        High-pass cutoff (in Hz)
    device : str
        Either 'cuda' (GPU) or 'cpu'; only used for rendering        
    """
    if isinstance(data, (str, Path)):
        # if data is a path to a hdf5 file, load it
        # (used by CLI)
        logger.info(f"Loading data from {data} ...")
        data = Data.load(data)

    v = data.v.reshape((data.v.shape[0], -1))

    nyq = 0.5 * data.fps  # sampling freq
    low = high_pass / nyq
    high = low_pass / nyq
    sos = butter(5, [low, high], analog=False, btype='band', output='sos')
    v = sosfilt(sos, v, axis=0)

    # Undo normalization to get data back on original scale
    data.v = v.reshape(data.v.shape).astype(np.float32)
    
    # Save!
    pth = data.path
    desc = 'desc-' + pth.split('desc-')[1].split('_')[0] + '+filt'
    f_out = pth.split('desc-')[0] + desc
    data.plot_data(f_out + '_qc.png', plot_motion=False, plot_pca=True, n_pca=3)
    data.render_video(f_out + '_shape.gif', device=device)
    data.save(f_out + '_shape.h5')

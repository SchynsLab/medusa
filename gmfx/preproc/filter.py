import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfilt, savgol_filter
from torch import triplet_margin_loss

from ..io import load_h5
from ..utils import get_logger

logger = get_logger()


def filter(data, low_pass, high_pass, video=None):
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
        data = load_h5(data)

    for attr in ['v']:#, 'motion']:
        d = getattr(data, attr)
        if d is None:
            continue
        
        if isinstance(d, pd.DataFrame):
            d = d.to_numpy()
        
        d = d.reshape((d.shape[0], -1))
        mu = d.mean(axis=0)
        d = d - mu

        nyq = 0.5 * data.sf # sampling freq
        low = high_pass / nyq
        high = low_pass / nyq
        sos = butter(5, [low, high], analog=False, btype='band', output='sos')
        d = sosfilt(sos, d, axis=0)
        
        # Undo normalization to get data back on original scale
        d_filt = (d + mu).reshape(getattr(data, attr).shape).astype(np.float32)
        
        if isinstance(getattr(data, attr), pd.DataFrame):
            d_filt = pd.DataFrame(d_filt, columns=getattr(data, attr).columns)
        
        setattr(data, attr, d_filt)

    # Save!
    pth = data.path
    desc = 'desc-' + pth.split('desc-')[1].split('_')[0] + '+filt'
    f_out = pth.split('desc-')[0] + desc
    data.plot_data(f_out + '_qc.png', plot_motion=True, plot_pca=True, n_pca=3)
    data.render_video(f_out + '_shape.gif', video=video)
    data.save(f_out + '_shape.h5')

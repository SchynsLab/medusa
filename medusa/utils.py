from curses.ascii import isdigit
import os
import sys
import logging
import numpy as np
from skimage.transform import rescale as rescale_skimage


logging.basicConfig(
    format="%(asctime)s [%(levelname)-7.7s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
    ],
)


def get_logger(level="INFO"):
    """Create a Python logger.

    Parameters
    ----------
    level : str
        Logging level ("INFO", "DEBUG", "WARNING")

    Returns
    -------
    logger : logging.Logger
        A Python logger

    Examples
    --------
    >>> logger = get_logger()
    >>> logger.info("Hello!")
    """
    logger = logging.getLogger("medusa")
        
    #if isinstance(level, str):
    #    if level.isdigit():
    #        level = int(level)

    #if isinstance(level, float):
    #    level = int(level)

    level = logging.getLevelName(level)
    logger.setLevel(level)
    return logger


def rescale(img, scale):
    
    img = rescale_skimage(img, scale, preserve_range=True, anti_aliasing=True, channel_axis=2)
    img = img.round().astype(np.uint8)
    
    return img


class hide_stdout:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
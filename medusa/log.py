import logging
from datetime import datetime

from tqdm import tqdm


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

    logging.basicConfig(
        format="%(asctime)s [%(levelname)-7.7s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("medusa")
    level = logging.getLevelName(level)
    logger.setLevel(level)
    return logger


def tqdm_log(iter_, logger, desc="Render shape"):
    """Creates an iterator with optional tqdm progressbar that plays nicely
    with an existing Medusa logger.

    Parameters
    ----------
    iter_ : iterable
        Initial iterable
    logger : logging.Logger
        Existing Medusa logger
    desc : str
        Text to display before progress bar

    Returns
    -------
    iter_ : iterable
        Iterable to be iterated over
    """
    if logger.level <= 20:
        pre = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")
        iter_ = tqdm(iter_, desc=f"{pre} {desc}")

    return iter_

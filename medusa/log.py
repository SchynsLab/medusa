import logging

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
    level = logging.getLevelName(level)
    logger.setLevel(level)
    return logger

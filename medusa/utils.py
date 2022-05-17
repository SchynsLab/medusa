import logging


def get_logger(verbose="INFO"):
    """ Create a Python logger.
    
    Parameters
    ----------
    verbose : str
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
        level=getattr(logging, verbose),
        format="%(asctime)s [%(levelname)-7.7s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("medusa")
    return logger

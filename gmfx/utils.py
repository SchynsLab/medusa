import tqdm
import logging


def get_logger(verbose='INFO'):
    
    logging.basicConfig(
        level=getattr(logging, verbose),
        format="%(asctime)s [%(levelname)-7.7s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
           logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger('gmfx')
    return logger
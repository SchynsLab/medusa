""" This module contains functions to load in "template data", i.e., the topological
templates used by the different models. 
"""

import yaml
import pickle
import trimesh
import warnings
from pathlib import Path


def get_template_mediapipe():
    """ Returns the template (vertices and triangles) of the canonical Mediapipe model. 
    
    Returns
    -------
    template : dict
        Dictionary with vertices ("v") and faces ("f")
        
    Examples
    --------
    Get the vertices and faces (triangles) of the standard Mediapipe topology (template):
    
    >>> template = get_template_mediapipe()
    >>> template['v'].shape
    (468, 3)
    >>> template['f'].shape
    (898, 3)
    """
    path = Path(__file__).parents[1] / 'data/mediapipe_template.obj'
    # Note to self: maintain_order=True is important, otherwise the
    # face order is all messed up
    with open(path, 'r') as f_in:
        data = trimesh.exchange.obj.load_obj(f_in, maintain_order=True)

    template = {'v': data['vertices'], 'f': data['faces']}
    return template


def get_template_flame():
    """ Returns the template (vertices and triangles) of the canonical Flame model. 
    
    Raises
    ------
    ValueError
        If the 'flame' package is not installed and/or the Flame file could not be found

    Returns
    -------
    template : dict
        Dictionary with vertices ("v") and faces ("f")    
        
    Examples
    --------
    Get the vertices and faces (triangles) of the standard Flame topology (template):

    >>> template = get_template_mediapipe()  # doctest: +SKIP
    >>> template['v'].shape  # doctest: +SKIP
    (468, 3)
    >>> template['f'].shape  # doctest: +SKIP
    (898, 3)
    """

    try:
        import flame
    except ImportError:
        raise ValueError("Package 'flame' is not installed; To install 'flame', check" 
                         "https://github.com/medusa-4D/flame")

    cfg_file = Path(flame.__file__).parent / 'data/config.yaml'
    with open(cfg_file, "r") as f_in:
        cfg = yaml.safe_load(f_in)
    
    flame_path = Path(cfg['flame_path'])
    if not flame_path.is_file():
        raise ValueError(f"Something wrong with 'flame' config file {cfg_file}; "
                         f"Could not find {flame_path}!")
        
    with open(flame_path, 'rb') as f_in:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            data = pickle.load(f_in, encoding='latin1')

    template = {'v': data['v_template'], 'f': data['f']}
    return template

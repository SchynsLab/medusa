""" This module contains functions to load in "template data", i.e., the topological
templates used by the different models. 
"""

import pickle
import trimesh
import warnings
import numpy as np
from pathlib import Path


def get_template_mediapipe():
    """ Returns the template (vertices and triangles) of the canonical Mediapipe model. 
    
    Returns
    -------
    template : dict
        Dictionary with vertices ("v") and faces ("f")    
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
    
    Returns
    -------
    template : dict
        Dictionary with vertices ("v") and faces ("f")    
    """
    path = Path(__file__).parents[2] / 'ext_data/FLAME/generic_model.pkl'
    if not path.is_file():
        raise ValueError(f"File {path} does not exist! Download the FLAME data as "
                          "explained here: https://lukas-snoek.com/medusa/getting_started/installation.html")
        
    with open(path, 'rb') as f_in:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            data = pickle.load(f_in, encoding='latin1')

    template = {'v': data['v_template'], 'f': data['f']}
    return template
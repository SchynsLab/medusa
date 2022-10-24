""" This module contains functions to load in "template data", i.e., the topological
templates used by the different models. 
"""

import trimesh
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


def get_template_flame(dense=False):
    """ Returns the template (vertices and triangles) of the canonical Flame model, in
    either its dense or coarse version. Note that this does exactly the same as the
    ``get_flame_template()`` function from the ``flame.data`` module.
    
    Parameters
    ----------
    dense : bool
        Whether to load in the dense version of the template (``True``) or the coarse
        version (``False``)
    
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
    Get the vertices and faces (triangles) of the standard Flame topology (template) in
    either the coarse version (``dense=False``) or dense version (``dense=True``)

    >>> template = get_template_flame(dense=False)  # doctest: +SKIP
    >>> template['v'].shape  # doctest: +SKIP
    (5023, 3)
    >>> template['f'].shape  # doctest: +SKIP
    (9976, 3)
    >>> template = get_template_flame(dense=True)  # doctest: +SKIP
    >>> template['v'].shape  # doctest: +SKIP
    (59315, 3)
    >>> template['f'].shape  # doctest: +SKIP
    (117380, 3)
    """

    try:
        from flame.data import get_template_flame
    except ImportError:
        raise ValueError("Package 'flame' is not installed; To install 'flame', check" 
                         "https://github.com/medusa-4D/flame")

    template = get_template_flame(dense)
    return template 
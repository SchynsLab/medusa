"""This module contains functions to load in "template data", i.e., the
topological templates used by the different models."""
from pathlib import Path
import numpy as np
import torch
import pickle
import h5py
import yaml

from ..io import load_obj
from ..defaults import DEVICE


def get_template_mediapipe(device=None):
    """Returns the template (vertices and triangles) of the canonical Mediapipe
    model.

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
    path = Path(__file__).parent / "mpipe/mediapipe_template.obj"
    template = load_obj(path)
    template = {key: template[key] for key in ['v', 'tris']}

    if device is not None:
        template['v'] = torch.as_tensor(template['v'], device=device).float()
        template['tris'] = torch.as_tensor(template['tris'], device=device).long()

    return template


def get_template_flame(topo='coarse', keys=None, device=None):
    """Returns the template (vertices and triangles) of the canonical Flame
    model, in either its dense or coarse version. Note that this does exactly
    the same as the ``get_flame_template()`` function from the ``flame.data``
    module.

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

    file = Path(__file__).parent / "flame/flame_template.h5"
    if not isinstance(topo, list):
        topo = [topo]

    template = {}
    with h5py.File(file, "r") as data:

        for topo_ in topo:
            template[topo_] = {}

            if keys is not None:
                for key in keys:
                    template[topo_][key] = data[topo_][key][:]
            else:
                for key in data[topo_].keys():
                    template[topo_][key] = data[topo_][key][:]

    if device is not None:

        for topo_ in template.keys():
            for key in template[topo_].keys():
                d = template[topo_][key]
                if d.dtype == np.uint32:
                    d = d.astype(np.int64)
                template[topo_][key] = torch.as_tensor(d, device=device)

    if len(topo) == 1:
        template = template[topo[0]]

    return template


def get_external_data_config(key=None):
    """Loads the FLAME config file (i.e., the yaml with paths to the FLAME-
    based models & data.

    Parameters
    ----------
    key : str
        If ``None`` (default), the config is returned as a dictionary;
        if ``str``, then the value associated with the key is returned

    Returns
    -------
    dict, str
        The config file as a dictionary if ``key=None``, else a string
        with the value associated with the key

    Examples
    --------
    Load in the entire config file as a dictionary

    >>> cfg = get_external_data_config()
    >>> isinstance(cfg, dict)
    True

    Get the path of the FLAME model:

    >>> flame_path = get_external_data_config(key='flame_path')
    """

    cfg_path = Path(__file__).parent / "config.yaml"
    if not cfg_path.is_file():
        # Fall back to default
        cfg_path = Path(__file__).parent / "default_config.yaml"

    with open(cfg_path, "r") as f_in:
        cfg = yaml.safe_load(f_in)

    for k, v in cfg.items():
        # If default location (~/.medusa_ext_data), resolve path
        cfg[k] = Path(v).expanduser()

    if key is None:
        return cfg
    else:
        if key not in cfg:
            raise ValueError(f"Key {key} not in config!")
        else:
            return Path(cfg[key])


def get_rigid_vertices(topo, device=DEVICE):
    """Gets the default 'rigid' vertices (i.e., vertices that can only move
    rigidly) for a given topology ('mediapipe', 'flame-coarse', 'flame-dense').

    Parameters
    ----------
    topo : str
        Topology name ('mediapipe', 'flame-coarse', or 'flame-dense')
    device : str
        Either 'cuda' (GPU) or 'cpu'

    Returns
    -------
    v_idx : torch.tensor
        A long tensor with indices of rigid vertices
    """

    if topo == 'mediapipe':
        v_idx = torch.tensor(
            [
                4, 6, 10, 33, 54, 67, 117, 119, 121, 127, 129, 132, 133, 136, 143, 147,
                198, 205, 263, 284, 297, 346, 348, 350, 356, 358, 361, 362, 365, 372,
                376, 420, 425
            ],
            dtype=torch.int64,
            device=device
        )
    elif topo == 'flame-coarse':
        path = get_external_data_config(key='flame_masks_path')
        with open(path, 'rb') as f_in:
            v_idx = torch.as_tensor(pickle.load(f_in, encoding='latin1')['scalp'],
                                    dtype=torch.int64, device=device)
    elif topo == 'flame-dense':
        v_idx = torch.ones(59315, dtype=torch.bool, device=device)
    else:
        raise ValueError(f"Unknown topology '{topo}'!")

    return v_idx


def get_vertex_template(topo, device=DEVICE):
    """Gets the default vertices (or 'template') for a given topology
    ('mediapipe', 'flame-coarse', 'flame-dense').

    Parameters
    ----------
    topo : str
        Topology name ('mediapipe', 'flame-coarse', or 'flame-dense')
    device : str
        Either 'cuda' (GPU) or 'cpu'

    Returns
    -------
    target : torch.tensor
        A float tensor with the default (template) vertices
    """

    if topo == 'mediapipe':
        target = get_template_mediapipe(device=device)['v']
    elif topo in ['flame-coarse', 'flame-dense']:
        target = get_template_flame(topo.split('-')[1], keys=['v'], device=device)['v']
    else:
        raise ValueError(f"Unknown topology '{topo}'!")

    return target


def get_tris(topo, device=DEVICE):
    """Gets the triangles for a given topology ('mediapipe', 'flame-coarse',
    'flame-dense').

    Parameters
    ----------
    topo : str
        Topology name ('mediapipe', 'flame-coarse', or 'flame-dense')
    device : str
        Either 'cuda' (GPU) or 'cpu'

    Returns
    -------
    tris : torch.tensor
        A long tensor with the triangles
    """
    if topo == 'mediapipe':
        tris = get_template_mediapipe(device=device)['tris']
    elif topo in ['flame-coarse', 'flame-dense']:
        tris = get_template_flame(topo.split('-')[1], keys=['tris'], device=device)['tris']
    else:
        raise ValueError(f"Unknown topology '{topo}'!")

    return tris.long()

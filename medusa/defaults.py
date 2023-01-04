"""Module with default objects, which values may depend on whether the system
has access to a GPU or not (such as ``DEVICE``)."""

from pathlib import Path

import cv2
import torch
import pickle

from .log import get_logger

# Set default device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""Default device ('cuda' or 'cpu') used across Medusa, which depends on whether
*cuda* is available ('cuda') or not ('cpu')."""

try:
    from .render.pytorch3d import PytorchRenderer as default_renderer
except ImportError:
    from .render.pyrender import PyRenderer as default_renderer

RENDERER = default_renderer
"""Default renderer used in Medusa, which depends on whether ``pytorch3d`` is installed
(in which case ``PytorchRenderer`` is used) or not (``PyRenderer`` is used)."""

FONT = str(Path(cv2.__path__[0]) / "qt/fonts/DejaVuSans.ttf")
"""Default font used in Medusa (DejaVuSans)."""

FLAME_MODELS = [
    "deca-coarse",
    "deca-dense",
    "emoca-coarse",
    "emoca-dense",
    "spectre-coarse",
    "spectre-dense",
]
"""Names of available FLAME-based models, which can be used when initializing a
``DecaReconModel``."""

RECON_MODELS = [
    "spectre-coarse",
    "emoca-dense",
    "emoca-coarse",
    "deca-dense",
    "deca-coarse",
    "mediapipe",
]
"""Names of all available reconstruction models."""

LOGGER = get_logger(level="INFO")
"""Default logger used in Medusa."""

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
    from .data import get_external_data_config

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
        from .data import get_template_mediapipe
        target = get_template_mediapipe(device=device)['v']
    elif topo in ['flame-coarse', 'flame-dense']:
        from .data import get_template_flame
        target = get_template_flame(topo.split('-')[1], keys=['v'], device=device)['v']
    else:
        raise ValueError(f"Unknown topology '{topo}'!")

    return target

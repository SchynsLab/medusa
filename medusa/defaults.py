"""Module with some constants and defaults."""
from pathlib import Path

import cv2
import torch

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

LOGGER = get_logger(level="INFO")
"""Default logger used in Medusa."""

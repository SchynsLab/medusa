"""Module with default objects, which values may depend on whether the system
has access to a GPU or not (such as ``DEVICE``)."""

from pathlib import Path

import torch

from .log import get_logger

# Set default device
if torch.cuda.is_available():
    DEVICE = 'cuda'
#elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
#    DEVICE = 'mps'
else:
    DEVICE = "cpu"
"""Default device ('cuda' or 'cpu') used across Medusa, which depends on whether
*cuda* is available ('cuda') or not ('cpu')."""

#torch.set_default_device(DEVICE)

FONT = str(Path(__file__).parent / "data/DejaVuSans.ttf")
"""Default font used in Medusa (DejaVuSans)."""

FLAME_MODELS = [
    "deca-coarse",
    "deca-dense",
    "emoca-coarse",
    "emoca-dense",
]
"""Names of available FLAME-based models, which can be used when initializing a
``DecaReconModel``."""

RECON_MODELS = [
    "emoca-dense",
    "emoca-coarse",
    "deca-dense",
    "deca-coarse",
    "mediapipe",
]
"""Names of all available reconstruction models."""

LOGGER = get_logger(level="INFO")
"""Default logger used in Medusa."""

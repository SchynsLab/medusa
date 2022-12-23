"""A module to globally set configurations."""
import os

import torch

torch.set_grad_enabled(False)
torch.autograd.set_detect_anomaly(False)

# Set pyopengl to 'egl' for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"

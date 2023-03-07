"""Medusa's top-level module; sets some global configurations."""

import torch

torch.set_grad_enabled(False)
torch.autograd.set_detect_anomaly(False)

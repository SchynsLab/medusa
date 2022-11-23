import os
import torch

# Set pyopengl to 'egl' for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Set default device
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
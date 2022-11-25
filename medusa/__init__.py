import os
import cv2
import torch
from pathlib import Path


# Set pyopengl to 'egl' for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Set default device
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

FONT = str(Path(cv2.__path__[0]) / 'qt/fonts/DejaVuSans.ttf')

import numpy as np
import os.path as op
from pathlib import Path

here = Path(__file__).parent.resolve()

# Face indices
FACES = {
    'coarse': np.load(here / 'data/faces_flame.npy'),
    'dense': np.load(here / 'data/faces_flame_dense.npy')
}

# right, left
EYES_CENTER = [4597, 4051]
EYES_INNER = [3638, 3835]
EYES_OUTER = [2419, 1142]

# from top (eye level) to bottom (tip of nose)
NOSE_RIDGE = [
    3516, 3518, 3561, 3548, 3521, 3501, 3508, 3526, 3564
]

EYES_NOSE = EYES_CENTER + EYES_INNER + EYES_OUTER + NOSE_RIDGE

SCALP = [
    # Meridian from center of scalp downwards the meridian to about ear-level
    3567, 3559, 3558, 3785, 3525, 3539, 3530, 3574, 3545, 3557, 3523, 3552, 3536, 3529, 3535, 3517
    # ... should add more vertices lateral to meridian of course
]

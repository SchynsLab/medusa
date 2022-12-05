import os
from pathlib import Path

import cv2
import numpy as np
import torch

from medusa.crop import LandmarkAlignCropModel
from medusa.data import get_example_frame
from medusa.recon.flame import MicaReconModel
from medusa.render import Renderer


def test_mica_recon():
    """Tests the MICA recon model."""

    if 'GITHUB_ACTIONS' in os.environ:
        return

    img = get_example_frame(load_numpy=True, device='cpu')
    crop_model = LandmarkAlignCropModel()
    img_crop, crop_mat = crop_model(img)

    # Check single image recon
    model = MicaReconModel(device='cpu')
    out = model(img_crop)
    assert(out['v'].shape == (1, 5023, 3))
    assert(out['mat'] is None)

    # Check visually
    cam_mat = np.eye(4)
    cam_mat[2, 3] = 1
    renderer = Renderer(viewport=(512, 512), smooth=False, cam_mat=cam_mat)

    img = renderer(out['v'].squeeze() * 8, model.get_tris())
    f_out = Path(__file__).parent / 'test_viz/test_mica.png'
    cv2.imwrite(str(f_out), img)

    # Check batch image recon
    out = model(torch.cat([img_crop, img_crop]))
    assert(out['v'].shape == (2, 5023, 3))

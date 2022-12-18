from pathlib import Path

import cv2
import pytest
import numpy as np

from medusa.crop import LandmarkAlignCropModel
from medusa.data import get_example_frame
from medusa.recon.flame import MicaReconModel
from medusa.render import Renderer

from conftest import _check_gha_compatible


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mica_recon(device):
    """Tests the MICA recon model."""

    if not _check_gha_compatible(device):
        return

    img = get_example_frame()
    crop_model = LandmarkAlignCropModel(device=device)
    out_crop = crop_model(img)

    # Check single image recon
    model = MicaReconModel(device=device)
    out = model(out_crop['imgs_crop'])
    assert(out['v'].shape == (1, 5023, 3))
    assert(out['mat'] is None)

    # Check visually
    cam_mat = np.eye(4)
    cam_mat[2, 3] = 1
    renderer = Renderer(viewport=(512, 512), shading='flat', cam_mat=cam_mat)

    v = out['v'] * 8
    img = renderer(v[0, ...], model.get_tris())
    f_out = Path(__file__).parent / 'test_viz/recon/test_mica.png'
    cv2.imwrite(str(f_out), img[:, :, [2, 1, 0]])

    # Check batch image recon
    out = model(out_crop['imgs_crop'].repeat(2, 1, 1, 1))
    assert(out['v'].shape == (2, 5023, 3))

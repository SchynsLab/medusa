from pathlib import Path

import pytest
from conftest import _check_gha_compatible

from medusa.constants import RENDERER
from medusa.crop import LandmarkAlignCropModel
from medusa.data import get_example_frame
from medusa.recon.flame import MicaReconModel


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mica_recon(device):
    """Tests the MICA recon model."""

    if not _check_gha_compatible(device):
        return

    img = get_example_frame()
    crop_model = LandmarkAlignCropModel(device=device)
    out_crop = crop_model(img)

    # Check single image recon
    model = MicaReconModel(device=device)
    out = model(out_crop["imgs_crop"])
    assert out["v"].shape == (1, 5023, 3)
    assert out["mat"] is None

    # Check visually
    cam_mat = model.get_cam_mat()
    renderer = RENDERER(
        viewport=(512, 512), shading="flat", cam_mat=cam_mat, device=device
    )

    v = out["v"]
    img = renderer(v[0, ...], model.get_tris())
    f_out = Path(__file__).parent / "test_viz/recon/test_mica.png"
    renderer.save_image(f_out, img)

    # Check batch image recon
    out = model(out_crop["imgs_crop"].repeat(2, 1, 1, 1))
    assert out["v"].shape == (2, 5023, 3)

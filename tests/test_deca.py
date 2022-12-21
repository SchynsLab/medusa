from pathlib import Path

import cv2
import numpy as np
import pytest

from medusa.containers import Data4D
from medusa.crop import LandmarkBboxCropModel
from medusa.data import get_example_video
from medusa.recon import DecaReconModel
from medusa.constants import RENDERER

from conftest import _check_gha_compatible


@pytest.mark.parametrize("name", ["deca", "emoca", "spectre"])
@pytest.mark.parametrize("type_", ["coarse", "dense"])
@pytest.mark.parametrize("no_crop_mat", [False, True])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_deca_recon(name, type_, no_crop_mat, device):
    """Generic test of DECA-based recon models."""
    if not _check_gha_compatible(device):
        return

    vid = get_example_video(return_videoloader=True, device=device)
    metadata = vid.get_metadata()

    if no_crop_mat:
        img_size = (224, 224)
    else:
        img_size = metadata["img_size"]

    crop_model = LandmarkBboxCropModel(device=device)
    model_name = f"{name}-{type_}"
    recon_model = DecaReconModel(name=model_name, img_size=img_size, device=device)

    img_batch = next(iter(vid))
    vid.close()

    out_crop = crop_model(img_batch)

    if no_crop_mat:
        out_crop["crop_mats"] = None

    out = recon_model(out_crop["imgs_crop"], out_crop["crop_mats"])

    if type_ == "coarse":
        expected_shape = (img_batch.shape[0], 5023, 3)
    else:
        expected_shape = (img_batch.shape[0], 59315, 3)

    assert out["v"].shape == expected_shape
    assert out["mat"].shape == (img_batch.shape[0], 4, 4)

    if not no_crop_mat:
        cam_mat = np.eye(4)
        cam_mat[2, 3] = 4
        renderer = RENDERER(viewport=img_size, shading="flat", cam_mat=cam_mat, device=device)
        img = renderer(out["v"][0], recon_model.get_tris())
        f_out = Path(__file__).parent / f"test_viz/recon/test_{model_name}.png"
        renderer.save_image(f_out, img)
        renderer.close()

    if not no_crop_mat:
        # Only render when recon full image
        tris = recon_model.get_tris()
        cam_mat = recon_model.get_cam_mat()
        data = Data4D(video_metadata=metadata, tris=tris, cam_mat=cam_mat, **out)
        f_out = str(f_out).replace(".png", ".mp4")
        data.render_video(f_out, shading="flat", video=get_example_video())

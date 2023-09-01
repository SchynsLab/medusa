from pathlib import Path

import torch
import numpy as np
import pytest
from conftest import _is_device_compatible

from medusa.defaults import DEVICE
from medusa.crop import BboxCropModel
from medusa.data import get_example_image, get_example_data4d
from medusa.recon import DecaReconModel, Mediapipe
from medusa.render import PytorchRenderer


@pytest.mark.parametrize("shading", ["flat", "smooth"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shading(shading, device):
    if not _is_device_compatible(device):
        return

    data = get_example_data4d(load=True, model="mediapipe", device=device)
    viewport = data.video_metadata["img_size"]
    renderer = PytorchRenderer(
        viewport, cam_mat=data.cam_mat, cam_type="perspective", shading=shading,
        device=device
    )

    img = renderer(data.v[0], data.tris)

    assert img.shape[-3] == viewport[1]  # height
    assert img.shape[-2] == viewport[0]  # width

    img_orig = get_example_image(device=device, dtype=torch.uint8)
    img = renderer.alpha_blend(img, img_orig)
    f_out = (
        Path(__file__).parent
        / f"test_viz/render/render_shading-{shading}.jpg"
    )
    renderer.save_image(f_out, img)


@pytest.mark.parametrize("n_faces", [2, 3, 4])
@pytest.mark.parametrize("recon_model_name", ["mediapipe", "emoca-coarse"])
def test_multiple_faces(n_faces, recon_model_name):
    img = get_example_image(n_faces)
    n_exp = n_faces
    viewport = (img.shape[3], img.shape[2])

    if recon_model_name == "mediapipe":
        cam_type = "perspective"
        recon_model = Mediapipe(static_image_mode=True)
        inputs = {"imgs": img}
        cam_mat = None
    else:
        crop_model = BboxCropModel()
        out_crop = crop_model(img)
        recon_model = DecaReconModel("emoca-coarse", orig_img_size=viewport)
        inputs = {"crop_mat": out_crop["crop_mat"], "imgs": out_crop["imgs_crop"]}
        cam_type = "orthographic"
        cam_mat = np.eye(4)
        cam_mat[2, 3] = 4.0

    out = recon_model(**inputs)

    renderer = PytorchRenderer(viewport, cam_type=cam_type, cam_mat=cam_mat, shading="flat")
    img_recon = renderer(out["v"], recon_model.get_tris())
    img_final = renderer.alpha_blend(img_recon, img)

    f_out = (
        Path(__file__).parent
        / f"test_viz/render/render_model-{str(recon_model)}_exp-{n_exp}.jpg"
    )
    renderer.save_image(f_out, img_final)
    recon_model.close()

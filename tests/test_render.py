import os
from pathlib import Path

import numpy as np
import pytest
from conftest import _is_gha_compatible

from medusa.defaults import DEVICE
from medusa.crop import BboxCropModel
from medusa.data import get_example_image, get_example_data4d, get_example_video
from medusa.recon import DecaReconModel, Mediapipe, videorecon
from medusa.render import PytorchRenderer, VideoRenderer


@pytest.mark.parametrize("shading", ["flat", "smooth"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shading(shading, device):
    if not _is_gha_compatible(device):
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

    img_orig = get_example_image(load_numpy=True)
    img = renderer.alpha_blend(img, img_orig)
    f_out = (
        Path(__file__).parent
        / f"test_viz/render/renderer-{str(renderer)}_shading-{shading}.jpg"
    )
    renderer.save_image(f_out, img)



@pytest.mark.parametrize("n_faces", [2, 3, 4])
@pytest.mark.parametrize("recon_model_name", ["mediapipe", "emoca-coarse"])
def test_multiple_faces(n_faces, recon_model_name):
    img = get_example_image(n_faces)

    img = PytorchRenderer.load_image(img)
    viewport = (img.shape[1], img.shape[0])

    if recon_model_name == "mediapipe":
        cam_type = "perspective"
        recon_model = Mediapipe(static_image_mode=True)
        inputs = {"imgs": img.copy()}
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
        / f"test_viz/render/model-{str(recon_model)}_renderer-{str(renderer)}_exp-{n_exp}.jpg"
    )
    renderer.save_image(f_out, img_final)
    renderer.close()
    recon_model.close()


@pytest.mark.parametrize("n_faces", [1, 4])
def test_render_video(n_faces, device=DEVICE):
    if 'GITHUB_ACTIONS' in os.environ:
        # Too slow for Github Actions
        return

    vid = get_example_video(n_faces)
    data = get_example_data4d(n_faces, load=True, model='emoca-coarse', device=device)
    data.apply_vertex_mask('face')
    f_out = (
        Path(__file__).parent
        / f"test_viz/render/f{n_faces}_face.mp4"
    )
    renderer = VideoRenderer()
    renderer(f_out, data, video=vid)

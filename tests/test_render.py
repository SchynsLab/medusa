from pathlib import Path

import numpy as np
import pytest
from conftest import _is_gha_compatible

from medusa.defaults import DEVICE
from medusa.crop import LandmarkBboxCropModel
from medusa.data import get_example_frame, get_example_h5
from medusa.recon import DecaReconModel, Mediapipe, videorecon
from medusa.render import PyRenderer

try:
    from medusa.render import PytorchRenderer
except ImportError:
    renderers = [PytorchRenderer]
else:
    renderers = [PytorchRenderer, PyRenderer]


@pytest.mark.parametrize("shading", ["flat", "wireframe", "smooth"])
@pytest.mark.parametrize("Renderer", renderers)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shading(shading, Renderer, device):
    if not _is_gha_compatible(device):
        return

    if Renderer != PyRenderer and shading == "wireframe":
        return

    if Renderer == PyRenderer and device == "cuda":
        return

    data = get_example_h5(load=True, model="mediapipe", device=device)
    viewport = data.video_metadata["img_size"]
    renderer = Renderer(
        viewport, cam_type="perspective", shading=shading, device=device
    )

    img = renderer(data.v[0], data.tris)

    assert img.shape[-3] == viewport[1]  # height
    assert img.shape[-2] == viewport[0]  # width

    img_orig = get_example_frame(load_numpy=True)
    img = renderer.alpha_blend(img, img_orig)
    f_out = (
        Path(__file__).parent
        / f"test_viz/render/renderer-{str(renderer)}_shading-{shading}.jpg"
    )
    renderer.save_image(f_out, img)


@pytest.mark.parametrize("color", ["red", "blue"])
@pytest.mark.parametrize("width", [None, 3])
def test_pyrender_wireframe(color, width):
    data = get_example_h5(load=True, model="mediapipe", device=DEVICE)
    viewport = data.video_metadata["img_size"]
    c = None if color == "red" else (0, 0, 1, 1)
    renderer = PyRenderer(
        viewport,
        cam_type="perspective",
        shading="wireframe",
        wireframe_opts={"color": c, "width": width},
    )
    img = renderer(data.v[0], data.tris)
    f_out = (
        Path(__file__).parent
        / f"test_viz/render/wireframecolor-{color}_wireframewidth-{width}.jpg"
    )
    renderer.save_image(f_out, img)


@pytest.mark.parametrize("imgs_test", [2, 3, 4], indirect=True)
@pytest.mark.parametrize("Renderer", renderers)
@pytest.mark.parametrize("recon_model_name", ["mediapipe", "emoca-coarse"])
def test_multiple_faces(imgs_test, Renderer, recon_model_name):
    img, n_exp = imgs_test
    img = Renderer.load_image(img)
    viewport = (img.shape[1], img.shape[0])

    if recon_model_name == "mediapipe":
        cam_type = "perspective"
        recon_model = Mediapipe(static_image_mode=True)
        inputs = {"imgs": img.copy()}
        cam_mat = None
    else:
        crop_model = LandmarkBboxCropModel()
        out_crop = crop_model(img)
        recon_model = DecaReconModel("emoca-coarse", img_size=viewport)
        inputs = {"crop_mats": out_crop["crop_mats"], "imgs": out_crop["imgs_crop"]}
        cam_type = "orthographic"
        cam_mat = np.eye(4)
        cam_mat[2, 3] = 4.0

    out = recon_model(**inputs)

    if isinstance(recon_model, DecaReconModel):
        out["v"] = recon_model.apply_mask("face", out["v"])

    renderer = Renderer(viewport, cam_type=cam_type, cam_mat=cam_mat, shading="flat")
    img_recon = renderer(out["v"], recon_model.get_tris())
    img_final = renderer.alpha_blend(img_recon, img)

    f_out = (
        Path(__file__).parent
        / f"test_viz/render/model-{str(recon_model)}_renderer-{str(renderer)}_exp-{n_exp}.jpg"
    )
    renderer.save_image(f_out, img_final)
    renderer.close()
    recon_model.close()


@pytest.mark.parametrize("video_test", [1, 4], indirect=True)
@pytest.mark.parametrize("renderer", ["pyrender", "pytorch3d"])
def test_render_video(video_test, renderer, device=DEVICE):
    data = videorecon(video_test, "emoca-coarse", device=device, mask="face")
    f_out = (
        Path(__file__).parent
        / f"test_viz/render/renderer-{renderer}_{video_test.stem}.mp4"
    )
    data.render_video(f_out, renderer=renderer, video=video_test)


# def test_render_video_ultimate():
#     video_test = "./two_face_demo.mp4"
#     data = videorecon(video_test, "emoca-coarse", device="cpu", mask="face")
#     f_out = "./ultimate.mp4"
#     data.render_video(f_out, renderer="pyrender", video=video_test)

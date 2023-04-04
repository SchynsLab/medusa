import os
from pathlib import Path

import pytest
from conftest import _is_gha_compatible

from medusa.render import VideoRenderer
from medusa.containers import Data4D
from medusa.data import get_example_video, get_example_image
from medusa.recon import DecaReconModel
from medusa.crop import BboxCropModel
from medusa.render import PytorchRenderer


@pytest.mark.parametrize("n_faces", [1, 2, 3, 4])
def test_deca_recon_img(n_faces):
    """Test DECA-based recon models with single image."""
    img = get_example_image(n_faces)
    img = PytorchRenderer.load_image(img)
    deca_recon_model = DecaReconModel("emoca-coarse", orig_img_size=(img.shape[1], img.shape[0]))

    out = deca_recon_model(img)
    cam_mat = deca_recon_model.get_cam_mat()
    tris = deca_recon_model.get_tris()

    renderer = PytorchRenderer((img.shape[1], img.shape[0]), cam_mat, cam_type='orthographic', shading="smooth")

    img_r = renderer(out["v"], tris)
    img_r = renderer.alpha_blend(img_r, img)
    renderer.save_image(Path(__file__).parent / f'test_viz/recon/test_emoca-coarse_exp-{n_faces}.png', img_r)
    renderer.close()


@pytest.mark.parametrize("name", ["deca", "emoca", "spectre"])
@pytest.mark.parametrize("type_", ["coarse", "dense"])
@pytest.mark.parametrize("already_cropped", [False, True])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_deca_recon(name, type_, already_cropped, device):
    """Generic test of DECA-based recon models."""
    if not _is_gha_compatible(device):
        return

    vid = get_example_video(return_videoloader=True, device=device)
    metadata = vid.get_metadata()

    model_name = f"{name}-{type_}"
    recon_model = DecaReconModel(name=model_name, device=device)

    img_batch = next(iter(vid))
    vid.close()

    if already_cropped:
        crop_model = BboxCropModel(device=device)
        img_batch = crop_model(img_batch)["imgs_crop"]

    out = recon_model(img_batch.clone())

    if type_ == "coarse":
        expected_shape = (img_batch.shape[0], 5023, 3)
    else:
        expected_shape = (img_batch.shape[0], 59315, 3)

    assert out["v"].shape == expected_shape
    assert out["mat"].shape == (img_batch.shape[0], 4, 4)

    img_size = (224, 224) if already_cropped else metadata['img_size']
    cam_mat = recon_model.get_cam_mat()
    renderer = PytorchRenderer(
        viewport=img_size, shading="flat", cam_type='orthographic', cam_mat=cam_mat, device=device
    )
    img = renderer(out["v"][0], recon_model.get_tris())
    img = renderer.alpha_blend(img=img, background=img_batch[0])
    f_out = Path(__file__).parent / f"test_viz/recon/test_{model_name}_cropped-{already_cropped}.png"
    renderer.save_image(f_out, img)
    renderer.close()

    if not already_cropped and not 'GITHUB_ACTIONS' in os.environ:
        # Only render when recon full image
        tris = recon_model.get_tris()
        data = Data4D(video_metadata=metadata, tris=tris, cam_mat=cam_mat, **out, device=device)
        data.video_metadata['n_img'] = vid.batch_size  # avoid rendering full video
        f_out = str(f_out).replace(".png", ".mp4")
        renderer = VideoRenderer()
        renderer(f_out, data, video=get_example_video(device=device))

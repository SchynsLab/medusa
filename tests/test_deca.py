import os
from pathlib import Path

import torch
import pytest
from conftest import _is_gha_compatible
from torchvision.utils import save_image

from medusa.render import VideoRenderer
from medusa.containers import Data4D
from medusa.data import get_example_video, get_example_image
from medusa.recon import DecaReconModel
from medusa.crop import BboxCropModel
from medusa.render import PytorchRenderer
from medusa.defaults import DEVICE


@torch.inference_mode()
@pytest.mark.parametrize("n_faces", [2, 1, 3, 4])
def test_deca_recon_img(n_faces):
    """Test DECA-based recon models with single image."""
    img = get_example_image(n_faces)
    viewport = (img.shape[3], img.shape[2])
    deca_recon_model = DecaReconModel("emoca-coarse", orig_img_size=viewport)
    crop_model = BboxCropModel()
    crop_results = crop_model(img)
    img_crop, crop_mat = crop_results["imgs_crop"], crop_results["crop_mat"]

    out = deca_recon_model(img_crop, crop_mat)
    cam_mat = deca_recon_model.get_cam_mat()
    tris = deca_recon_model.get_tris()

    renderer = PytorchRenderer(viewport, cam_mat, cam_type='orthographic', shading="smooth")

    img_r = renderer(out["v"], tris)
    img_r = renderer.alpha_blend(img_r, img)
    renderer.save_image(Path(__file__).parent / f'test_viz/recon/test_emoca-coarse_exp-{n_faces}.png', img_r)
    save_image(img_crop.float(), Path(__file__).parent / f'test_viz/recon/test_emoca-coarse_exp-{n_faces}_cropped.jpg', normalize=True)
    exit()


@torch.inference_mode()
@pytest.mark.parametrize("name", ["deca", "emoca"])
@pytest.mark.parametrize("type_", ["coarse", "dense"])
@pytest.mark.parametrize("already_cropped", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_deca_recon(name, type_, already_cropped, device):
    """Generic test of DECA-based recon models."""
    if not _is_gha_compatible(device):
        return

    vid = get_example_video(return_videoloader=True, device=device)
    metadata = vid.get_metadata()

    model_name = f"{name}-{type_}"
    recon_model = DecaReconModel(name=model_name, device=device)

    img_batch = next(iter(vid)).to(device)

    if already_cropped:
        crop_model = BboxCropModel(device=device)
        img_batch = crop_model(img_batch)["imgs_crop"]

    out = recon_model(img_batch)

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
    vid.close()

    if not already_cropped and not 'GITHUB_ACTIONS' in os.environ and device == 'cuda':
        # Only render when recon full image and device is 'cuda' otherwise takes forever
        tris = recon_model.get_tris()
        data = Data4D(video_metadata=metadata, tris=tris, cam_mat=cam_mat, **out, device=device)
        data.video_metadata['n_img'] = vid.batch_size  # avoid rendering full video
        f_out = str(f_out).replace(".png", ".mp4")
        renderer = VideoRenderer()
        renderer.render(f_out, data, video=get_example_video(device=device))

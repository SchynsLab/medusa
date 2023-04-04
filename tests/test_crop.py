import os
from pathlib import Path

import pytest
from conftest import _is_gha_compatible

from medusa.containers.results import BatchResults
from medusa.crop import AlignCropModel, BboxCropModel
from medusa.data import get_example_image, get_example_video


@pytest.mark.parametrize("Model", [AlignCropModel, BboxCropModel])
@pytest.mark.parametrize("lm_name", ["2d106det", "1k3d68"])
@pytest.mark.parametrize("n_faces", [0, 1, 2, 3, 4, [0, 1], [0, 1, 2], [0, 1, 2, 3, 4]])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_crop_model(Model, lm_name, n_faces, device):
    """Generic tests for crop models."""
    if not _is_gha_compatible(device):
        return

    imgs = get_example_image(n_faces)
    if isinstance(n_faces, int):
        n_exp = n_faces
    else:
        n_exp = sum(n_faces)

    if Model == BboxCropModel:
        model = Model(lm_name, (224, 224), device=device)
    else:
        if lm_name == "2d106det":
            # not necessary to test
            return

        model = Model((112, 112), device=device)

    out_crop = model(imgs)
    out_crop = BatchResults(device=device, **out_crop)

    template = getattr(model, "template", None)
    if Model == BboxCropModel:
        f_out = (
            Path(__file__).parent
            / f"test_viz/crop/{str(model)}_lm-{lm_name}_exp-{n_exp}.jpg"
        )
    else:
        f_out = Path(__file__).parent / f"test_viz/crop/{str(model)}_exp-{n_exp}.jpg"

    if 'GITHUB_ACTIONS' in os.environ:
        # Too slow for Github Actions
        return

    out_crop.visualize(f_out, imgs, template=template)


@pytest.mark.parametrize("Model", [AlignCropModel, BboxCropModel])
@pytest.mark.parametrize("n_faces", [0, 1, 2, 3, 4])
def test_crop_model_vid(Model, n_faces):
    """Test of crop model applied to videos and the visualization thereof."""

    video_test = get_example_video(n_faces)

    if Model == BboxCropModel:
        crop_size = (224, 224)
        model = Model("2d106det", crop_size)
    else:
        crop_size = (448, 448)
        model = Model(crop_size)

    results = model.crop_faces_video(video_test, save_imgs=True)

    if getattr(results, "lms", None) is None:
        return

    results.sort_faces(attr="lms")

    if 'GITHUB_ACTIONS' in os.environ:
        # Too slow for Github Actions
        return

    f_out = Path(__file__).parent / f"test_viz/crop/{str(model)}_{video_test.stem}.mp4"
    template = getattr(model, "template", None)
    results.visualize(
        f_out,
        results.imgs,
        template=template,
        video=True,
        crop_size=crop_size,
        show_cropped=True,
    )

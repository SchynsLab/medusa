from pathlib import Path

import pytest
from conftest import _check_gha_compatible

from medusa.crop import LandmarkAlignCropModel, LandmarkBboxCropModel
from medusa.containers.results import BatchResults


@pytest.mark.parametrize('Model', [LandmarkAlignCropModel, LandmarkBboxCropModel])
@pytest.mark.parametrize('lm_name', ['2d106det', '1k3d68'])
@pytest.mark.parametrize('imgs_test', [0, 1, 2, 3, 4, [0, 1], [0, 1, 2], [0, 1, 2, 3, 4]], indirect=True)
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_crop_model(Model, lm_name, imgs_test, device):
    """Generic tests for crop models."""
    if not _check_gha_compatible(device):
        return

    imgs, n_exp = imgs_test

    if Model == LandmarkBboxCropModel:
        model = Model(lm_name, (224, 224), device=device)
    else:
        if lm_name == '2d106det':
            # not necessary to test
            return

        model = Model((112, 112), device=device)

    out_crop = model(imgs)
    out_crop = BatchResults(device=device, **out_crop)

    template = getattr(model, 'template', None)
    if Model == LandmarkBboxCropModel:
        f_out = Path(__file__).parent / f'test_viz/crop/{str(model)}_lm-{lm_name}_exp-{n_exp}.jpg'
    else:
        f_out = Path(__file__).parent / f'test_viz/crop/{str(model)}_exp-{n_exp}.jpg'

    out_crop.visualize(f_out, imgs, template=template)


@pytest.mark.parametrize('Model', [LandmarkAlignCropModel, LandmarkBboxCropModel])
@pytest.mark.parametrize('video_test', [0, 1, 2, 3, 4], indirect=True)
def test_crop_model_vid(Model, video_test):
    """Test of crop model applied to videos and the visualization thereof."""
    if Model == LandmarkBboxCropModel:
        crop_size = (224, 224)
        model = Model('2d106det', crop_size)
    else:
        crop_size = (448, 448)
        model = Model(crop_size)

    results = model.crop_faces_video(video_test, save_imgs=True)

    if getattr(results, 'lms', None) is None:
        return

    results.sort_faces(attr='lms')

    f_out = Path(__file__).parent / f'test_viz/crop/{str(model)}_{video_test.stem}.mp4'
    template = getattr(model, 'template', None)
    results.visualize(f_out, results.imgs, template=template, video=True, crop_size=crop_size, show_cropped=True)

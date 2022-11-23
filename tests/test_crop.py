import os
import pytest
from pathlib import Path
from medusa.data import get_example_frame
from medusa.crop import LandmarkAlignCropModel, LandmarkBboxCropModel

from test_utils import _check_gha_compatible

imgs = ['no_face.jpg', 'one_face.jpg', 'two_faces.jpg', 'three_faces.jpg']

@pytest.mark.parametrize('Model', [LandmarkAlignCropModel, LandmarkBboxCropModel])
@pytest.mark.parametrize('lm_name', ['2d106det', '1k3d68'])
@pytest.mark.parametrize('output_size', [(112, 112), (224, 224)])
@pytest.mark.parametrize('img_params', zip(imgs, [None, 1, 2, 3]))
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_align_crop_model(Model, lm_name, output_size, img_params, batch_size, device):

    if not _check_gha_compatible(device):
        return

    if Model == LandmarkBboxCropModel:
        model = Model(lm_name, output_size, device=device, return_lmk=True)
    else:
        if lm_name == '2d106det':
            return

        model = Model(output_size, device=device, return_lmk=True)
    
    img, exp_n_face = img_params
    img_path = Path(__file__).parent / f'test_data/detection/{img}'
    img_path = batch_size * [img_path]
    
    img_crop, crop_mat, lmk = model(img_path)
    assert(len(img_crop) == len(crop_mat) == batch_size)

    if exp_n_face is None:
        for output in (img_crop, crop_mat, lmk):
            assert(all(outp is None for outp in output))
        
    else:
        assert(img_crop[0].shape[0] == crop_mat[0].shape[0])
        n_detected = img_crop[0].shape[0]
        assert(n_detected == exp_n_face)

    if batch_size == 1:
        if Model == LandmarkBboxCropModel:
            f_out = Path(__file__).parent / f'test_viz/crop/{str(model)}_lm-{lm_name}_size-{output_size[0]}_{img}'    
        else:
            f_out = Path(__file__).parent / f'test_viz/crop/{str(model)}_size-{output_size[0]}_{img}'
    
        model.visualize(img_crop, lmk, f_out=f_out)

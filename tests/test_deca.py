import os
import cv2
import pytest
import numpy as np
from pathlib import Path
from medusa.data import get_example_video
from medusa.crop import FanCropModel
from medusa.recon import DecaReconModel
from medusa.render import Renderer
from medusa.core import Flame4D


@pytest.mark.parametrize("name", ['deca', 'emoca', 'spectre'])
@pytest.mark.parametrize("type_", ['coarse', 'dense'])
@pytest.mark.parametrize("no_crop_mat", [False, True])
def test_deca_recon(name, type_, no_crop_mat):

    device = 'cuda'

    model_name = f'{name}-{type_}'

    vid = get_example_video(return_videoloader=True, device=device)
    metadata = vid.get_metadata()

    if no_crop_mat:
        img_size = (224, 224)
    else:
        img_size = metadata['img_size']

    crop_model = FanCropModel(device=device)
    recon_model = DecaReconModel(name=model_name, img_size=img_size, device=device)

    img_batch = next(vid)
    img_crop, crop_mat = crop_model(img_batch)
    
    if not no_crop_mat:
        recon_model.crop_mat = crop_mat

    out = recon_model(img_crop)

    if type_ == 'coarse':
        expected_shape = (img_batch.shape[0], 5023, 3)
    else:
        expected_shape = (img_batch.shape[0], 59315, 3)

    assert(out['v'].shape == expected_shape)
    assert(out['mat'].shape == (img_batch.shape[0], 4, 4))

    cam_mat = np.eye(4)
    cam_mat[2, 3] = 4
    renderer = Renderer(viewport=img_size, smooth=False, cam_mat=cam_mat)
    img = renderer(out['v'][0, ...], recon_model.get_tris())
    
    f_out = Path(__file__).parent / f'test_viz/test_{model_name}.png'
    cv2.imwrite(str(f_out), img)

    recon_model.close()
    vid.close()
    
    if not no_crop_mat:
        # Only render when recon full image
        kwargs = {**out, **vid.get_metadata()}
        data = Flame4D(recon_model=recon_model, f=recon_model.get_tris(), **kwargs)
        f_out = str(f_out).replace('.png', '.mp4')
        data.render_video(f_out, smooth=False, video=get_example_video())

import os
import pytest
from medusa.data import get_example_video
from medusa.crop import FanCropModel
from medusa.recon import DecaReconModel


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_deca_recon(device):

    if 'GITHUB_ACTIONS' in os.environ and device == 'cuda':
        return

    vid = get_example_video(return_videoloader=True, device=device)
    metadata = vid.get_metadata()
    img_size = metadata['img_size']

    crop_model = FanCropModel(device=device)
    recon_model = DecaReconModel(name='deca-coarse', img_size=img_size, device=device)

    img_batch = next(vid)
    img_crop, crop_mat = crop_model(img_batch)
    recon_model.crop_mat = crop_mat
    out = recon_model(img_crop)
    assert(out['v'].shape == (img_batch.shape[0], 5023, 3))
    assert(out['mat'].shape == (img_batch.shape[0], 4, 4))

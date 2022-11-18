import os
import pytest
from pathlib import Path
from medusa.io import VideoLoader
from medusa.data import get_example_video


@pytest.mark.parametrize("ext", ['.mp4', '.avi'])
def test_videoloader_ext(ext):

    vid = Path(__file__).parents[1] / f'test_data/videoloader/example_vid{ext}'
    loader = VideoLoader(vid, device='cpu', loglevel='WARNING')


def test_videoloader_full_iteration():

    vid = get_example_video(return_videoloader=False)
    loader = VideoLoader(vid, device='cpu', loglevel='WARNING')
    
    n_expected = len(loader)
    n_actual = 0
    for img_batch in loader:
        n_actual += img_batch.shape[0]

    assert(n_actual == n_expected)


@pytest.mark.parametrize("rescale_factor", [None, 0.5])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_videoloader_params(rescale_factor, batch_size, device):

    if 'GITHUB_ACTIONS' in os.environ and device == 'cuda':
        return
    
    vid = get_example_video(return_videoloader=False)
    loader = VideoLoader(
        vid, rescale_factor=rescale_factor, n_preload=512,
        device=device, batch_size=batch_size,
        loglevel='WARNING'
    )

    metadata = loader.get_metadata()
    img_batch = next(loader)

    assert(img_batch.shape[0] == batch_size)
    assert(img_batch.device.type == device)
    assert(img_batch.shape[1:3] == metadata['img_size'][::-1])
import os
from pathlib import Path

import pytest

from medusa.data import get_example_video
from medusa.io import VideoLoader
from test_utils import _check_gha_compatible


@pytest.mark.parametrize("ext", ['.mp4', '.avi'])
def test_videoloader_ext(ext):
    vid = Path(__file__).parent / f'test_data/example_vid{ext}'
    loader = VideoLoader(vid, device='cpu', loglevel='WARNING')


def test_videoloader_full_iteration():
    vid = get_example_video(return_videoloader=False)
    loader = VideoLoader(vid, device='cpu', loglevel='WARNING')

    n_expected = len(loader)
    n_actual = 0
    for img_batch in loader:
        n_actual += img_batch.shape[0]

    assert(n_actual == n_expected)


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_videoloader_params(batch_size, device):
    if not _check_gha_compatible(device):
        return

    vid = get_example_video(return_videoloader=False)
    loader = VideoLoader(
        vid, device=device, batch_size=batch_size, loglevel='WARNING'
    )

    metadata = loader.get_metadata()
    img_batch = next(iter(loader))

    assert(img_batch.shape[0] == batch_size)
    assert(img_batch.device.type == device)
    assert(img_batch.shape[1:3] == metadata['img_size'][::-1])

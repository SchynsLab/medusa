import os
import pytest
from pathlib import Path


@pytest.fixture
def imgs_test(request):
    nr_faces = request.param

    if isinstance(nr_faces, int):
        nr_faces = [nr_faces]

    data_dir = Path(__file__).parent / 'test_data'
    imgs = []
    for nr in nr_faces:
        imgs.append(data_dir / f'{nr}_face.jpg')

    if len(imgs) == 1:
        imgs = imgs[0]

    return imgs, sum(nr_faces)


@pytest.fixture
def video_test(request):
    nr_faces = request.param

    data_dir = Path(__file__).parent / 'test_data'
    vid = data_dir / f'{nr_faces}_face.mp4'
    return vid


def _check_gha_compatible(device):
    if device == 'cuda' and 'GITHUB_ACTIONS' in os.environ:
        return False
    else:
        return True
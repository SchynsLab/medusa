import torch
import pytest
import numpy as np
from pathlib import Path
from medusa.data import get_example_image, get_example_video, get_example_data4d
from medusa.data import get_template_flame, get_template_mediapipe
from medusa.io import VideoLoader
from medusa.containers import Data4D
from conftest import _is_gha_compatible


@pytest.mark.parametrize('n_faces', [None, 0, 1, 2, 3, 4, [1, 3], (2, 4)])
@pytest.mark.parametrize('load_torch', [True, False])
@pytest.mark.parametrize('load_numpy', [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_example_image(n_faces, load_torch, load_numpy, device):

    if not _is_gha_compatible(device):
        return

    if load_numpy and load_torch:
        return

    img = get_example_image(n_faces, load_numpy, load_torch, device=device)

    if isinstance(n_faces, (list, tuple)):
        assert(isinstance(img, list))
    else:
        img = [img]

    if not load_torch and not load_numpy:
        for img_ in img:
            assert(isinstance(img_, Path))

    if load_torch:
        for img_ in img:
            assert(torch.is_tensor(img_))
            assert(img_.device.type == device)

    if load_numpy:
        for img_ in img:
            assert(isinstance(img_, np.ndarray))


@pytest.mark.parametrize('n_faces', [None, 0, 1, 2, 3, 4])
@pytest.mark.parametrize('return_videoloader', [True, False])
@pytest.mark.parametrize('kwargs', [{}, {'batch_size': 16}])
def test_get_example_video(n_faces, return_videoloader, kwargs):

    out = get_example_video(n_faces, return_videoloader, **kwargs)

    if return_videoloader:
        assert(isinstance(out, VideoLoader))

        if kwargs:
            batch = next(iter(out))
            assert(batch.shape[0] == kwargs['batch_size'])


@pytest.mark.parametrize('n_faces', [None, 1, 2, 3, 4])
@pytest.mark.parametrize('load', [True, False])
@pytest.mark.parametrize('model', ['mediapipe', 'emoca-coarse'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_get_example_data4d(n_faces, load, model, device):

    if not _is_gha_compatible(device):
        return

    out = get_example_data4d(n_faces, load, model, device)

    if load:
        assert(isinstance(out, Data4D))
        assert(out.device == device)
        assert(out.v.device.type == device)

        if model == 'mediapipe':
            assert(out.v.shape[1] == 468)
        else:
            assert(out.v.shape[1] == 5023)
    else:
        assert(isinstance(out, Path))
        assert(out.is_file())


@pytest.mark.parametrize('topo', ['coarse', 'dense'])
@pytest.mark.parametrize('device', [None, 'cpu', 'cuda'])
def test_get_template_flame(topo, device):

    if not _is_gha_compatible(device):
        return

    out = get_template_flame(topo, device=device)

    if topo == 'dense':
        assert(out['v'].shape == (59315, 3))
        assert(out['tris'].shape == (117380, 3))
    else:
        assert(out['v'].shape == (5023, 3))
        assert(out['tris'].shape == (9976, 3))

    if device is not None:
        assert(out['v'].device.type == device)
        assert(out['tris'].device.type == device)


def test_get_template_mediapipe():

    out = get_template_mediapipe()
    assert(out['v'].shape == (468, 3))
    assert(out['tris'].shape == (898, 3))

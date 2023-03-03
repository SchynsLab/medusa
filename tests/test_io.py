import pytest
import medusa
import numpy as np
from pathlib import Path
from medusa.io import load_obj, save_obj
from medusa.data import get_template_flame


@pytest.mark.parametrize('device', [None, 'cpu'])
def test_load_obj(device):
    f = Path(medusa.__file__).parent / 'data/mpipe/mediapipe_template.obj'
    out = load_obj(f, device)

    for key in ['v', 'tris']:
        assert(key in out)
        if device is None:
            assert(isinstance(out[key], np.ndarray))
        else:
            assert(out[key].device.type == device)


@pytest.mark.parametrize('device', [None, 'cpu'])
def test_save_obj(device):
    template = get_template_flame(keys=['v', 'tris'], device=device)
    f = Path(__file__).parent / 'test_viz/io/flame_coarse.obj'
    save_obj(f, template)

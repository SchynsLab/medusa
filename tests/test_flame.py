import pytest
import torch
from pathlib import Path

from medusa.recon.flame.decoders import FlameShape, FlameLandmark
from medusa.defaults import DEVICE
from pytorch3d.renderer import PerspectiveCameras
from torchvision.utils import draw_keypoints, save_image


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('parameters', [None, ['shape'], ['shape', 'global_pose']])
def test_flame_shape_no_inputs(batch_size, parameters):

    flame = FlameShape(n_shape=300, n_expr=100, parameters=parameters)
    v, R = flame(batch_size=batch_size)

    assert(v.shape[0] == batch_size)
    assert(R.shape[0] == batch_size)

    if parameters is not None:
        assert(v.requires_grad)
        for param in parameters:
            assert(flame.get_parameter(param).requires_grad)
    else:
        assert(not v.requires_grad)
        for param in flame.parameters():
            assert(not param.requires_grad)


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('inputs', [{}, {'shape': torch.randn(1, 300, device=DEVICE)}])
@pytest.mark.parametrize('parameters', [None, ['shape'], ['global_pose']])
def test_flame_shape_with_inputs(batch_size, inputs, parameters):

    flame = FlameShape(n_shape=300, n_expr=100, parameters=parameters)
    if not inputs:
        batch_size = batch_size
    else:
        for key in inputs:
            inputs[key] = inputs[key].repeat(batch_size, 1)
        batch_size = None

    if 'shape' in inputs and parameters is not None:
        if 'shape' in parameters:
            return

    v, R = flame(batch_size=batch_size, **inputs)

    if batch_size is None:
        batch_size = next(iter(inputs.values())).shape[0]

    assert(v.shape[0] == batch_size)
    assert(R.shape[0] == batch_size)


def test_flame_shape_grad():

    flame = FlameShape(n_shape=300, n_expr=100, parameters=['shape'])
    optim = torch.optim.Adam([flame.shape], lr=0.1)
    for _ in range(10):
        v, R = flame(batch_size=1)
        loss = v.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()


@pytest.mark.parametrize('lm_type', ['68', 'mp'])
def test_flame_landmark(lm_type):
    flame = FlameShape()
    v, R = flame(batch_size=1)

    flame_lm = FlameLandmark(lm_type=lm_type)
    poses = flame.get_full_pose()
    lms = flame_lm(v, poses)

    T = torch.tensor([[0., 0., 1]], device=DEVICE)
    cam = PerspectiveCameras(focal_length=10., T=T, device=DEVICE)
    lms = cam.transform_points_screen(lms, image_size=(512, 512))
    img = torch.zeros(3, 512, 512, device=DEVICE, dtype=torch.uint8)
    img = draw_keypoints(img, lms, radius=3, colors='green')

    f_out = Path(__file__).parent / f'test_viz/flame/flame_landmark_{lm_type}.png'
    save_image(img.float(), f_out)

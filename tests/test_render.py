import cv2
import torch
import pytest
from pathlib import Path
from medusa.render import Renderer
from medusa.data import get_example_h5, get_example_frame
from medusa import DEVICE


@pytest.mark.parametrize('shading', ['flat', 'wireframe', 'smooth'])
def test_pyrender_shading(shading):
    data = get_example_h5(load=True, model='mediapipe', device=DEVICE)
    viewport = data.video_metadata['img_size']
    renderer = Renderer(viewport, cam_type='intrinsic', shading=shading)
    img = renderer(data.v[0], data.tris)
    assert(img.shape[0] == viewport[1])  # height
    assert(img.shape[1] == viewport[0])  # width

    img_orig = get_example_frame(load_numpy=True)
    img = renderer.alpha_blend(img, img_orig)

    f_out = Path(__file__).parent / f'test_viz/render/pyrender_shading-{shading}.jpg'
    cv2.imwrite(str(f_out), img=img[:, :, [2, 1, 0]])


@pytest.mark.parametrize('color', [None, (0, 0, 1, 1)])
@pytest.mark.parametrize('width', [None, 3])
def test_pyrender_wireframe(color, width):
    data = get_example_h5(load=True, model='mediapipe', device=DEVICE)
    viewport = data.video_metadata['img_size']
    renderer = Renderer(viewport, cam_type='intrinsic', shading='wireframe',
                        wireframe_opts={'color': color, 'width': width})
    img = renderer(data.v[0], data.tris)
    f_out = Path(__file__).parent / f'test_viz/render/pyrender_wireframecolor-{color}_wireframewidth-{width}.jpg'
    cv2.imwrite(str(f_out), img=img[:, :, [2, 1, 0]])


def test_pyrender_two_faces():
    data = get_example_h5(load=True, model='mediapipe', device=DEVICE)
    viewport = data.video_metadata['img_size']
    renderer = Renderer(viewport, cam_type='intrinsic', shading='flat')

    v = data.v[0].clone()
    v[..., 0] -= 8
    v = torch.stack([v, data.v[0]])
    img = renderer(v, data.tris)
    f_out = Path(__file__).parent / f'test_viz/render/pyrender_two_faces.jpg'
    cv2.imwrite(str(f_out), img=img[:, :, [2, 1, 0]])

import torch
import numpy as np
import pytest
from pathlib import Path
from medusa.render import PyRenderer
from medusa.data import get_example_h5, get_example_frame
from medusa.recon import Mediapipe, DecaReconModel
from medusa.crop import LandmarkBboxCropModel
from torchvision.utils import save_image
from conftest import _check_gha_compatible

from medusa import DEVICE

try:
    from medusa.render import PytorchRenderer
except ImportError:
    renderers = [PytorchRenderer]
else:
    renderers = [PytorchRenderer, PyRenderer]


# @pytest.mark.parametrize('shading', ['flat', 'wireframe', 'smooth'])
# @pytest.mark.parametrize('Renderer', renderers)
# @pytest.mark.parametrize('device', ['cpu', 'cuda'])
# def test_shading(shading, Renderer, device):

#     if not _check_gha_compatible(device):
#         return

#     if Renderer != PyRenderer and shading == 'wireframe':
#         return

#     if Renderer == PyRenderer and device == 'cuda':
#         return

#     data = get_example_h5(load=True, model='mediapipe', device=device)
#     viewport = data.video_metadata['img_size']
#     renderer = Renderer(viewport, cam_type='perspective', shading=shading, device=device)

#     img = renderer(data.v[0], data.tris)

#     assert(img.shape[-3] == viewport[1])  # height
#     assert(img.shape[-2] == viewport[0])  # width

#     img_orig = get_example_frame(load_numpy=True)
#     img = renderer.alpha_blend(img, img_orig)
#     f_out = Path(__file__).parent / f'test_viz/render/pyrender_renderer-{str(renderer)}_shading-{shading}.jpg'

#     if not torch.is_tensor(img):
#         img = torch.as_tensor(img, device=data.device)

#     renderer.save_image(f_out, img)


# @pytest.mark.parametrize('color', ['red', 'blue'])
# @pytest.mark.parametrize('width', [None, 3])
# def test_pyrender_wireframe(color, width):
#     data = get_example_h5(load=True, model='mediapipe', device=DEVICE)
#     viewport = data.video_metadata['img_size']
#     c = None if color == 'red' else (0, 0, 1, 1)
#     renderer = PyRenderer(viewport, cam_type='perspective', shading='wireframe',
#                           wireframe_opts={'color': c, 'width': width})
#     img = renderer(data.v[0], data.tris)
#     f_out = Path(__file__).parent / f'test_viz/render/pyrender_wireframecolor-{color}_wireframewidth-{width}.jpg'
#     renderer.save_image(f_out, img)


@pytest.mark.parametrize('imgs_test', [2, 3, 4], indirect=True)
@pytest.mark.parametrize('Renderer', renderers)
def test_multiple_faces(imgs_test, Renderer):
    img, n_exp = imgs_test
    img = Renderer.load_image(img)
    mp_model = Mediapipe(static_image_mode=True)
    out = mp_model(img.copy())

    viewport = (img.shape[1], img.shape[0])
    renderer = Renderer(viewport, cam_type='perspective', shading='flat')

    img_recon = renderer(out['v'], mp_model.get_tris())
    img_final = renderer.alpha_blend(img_recon, img)
    f_out = Path(__file__).parent / f'test_viz/render/pyrender_mediapipe_renderer-{str(renderer)}_exp-{n_exp}.jpg'
    renderer.save_image(f_out, img_final)

    renderer.close()

    crop_model = LandmarkBboxCropModel()
    out_crop = crop_model(img)
    emoca_model = DecaReconModel('emoca-coarse', img_size=viewport)
    out_recon = emoca_model(out_crop['imgs_crop'], out_crop['crop_mats'])
    v = emoca_model.apply_mask('face', out_recon['v'])

    renderer = Renderer(viewport, cam_type='orthographic', shading='flat', cam_mat=None)
    tris = emoca_model.get_tris()
    img_recon = renderer(v, tris)
    img_final = renderer.alpha_blend(img_recon, img)
    f_out = Path(__file__).parent / f'test_viz/render/pyrender_emoca_renderer-{str(renderer)}_exp-{n_exp}.jpg'
    renderer.save_image(f_out, img_final)

    renderer.close()

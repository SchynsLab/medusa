from pathlib import Path

import pytest
import torch
from test_utils import _check_gha_compatible

from medusa.crop import LandmarkAlignCropModel, LandmarkBboxCropModel
from medusa.crop.base import CropResults
from medusa.io import VideoLoader

# imgs = ['no_face', 'one_face', 'two_faces', 'three_faces',
#         ['no_face', 'one_face'], ['no_face', 'two_faces'],
#         ['one_face', 'two_faces', 'three_faces']]
# n_exp = [0, 1, 2, 3, 1, 2, 6]
# img_params = zip(imgs, n_exp)


# @pytest.mark.parametrize('Model', [LandmarkAlignCropModel, LandmarkBboxCropModel])
# @pytest.mark.parametrize('lm_name', ['2d106det', '1k3d68'])
# @pytest.mark.parametrize('img_params', img_params)
# @pytest.mark.parametrize('device', ['cpu', 'cuda'])
# def test_crop_model(Model, lm_name, img_params, device):

#     if not _check_gha_compatible(device):
#         return

#     if Model == LandmarkBboxCropModel:
#         model = Model(lm_name, (224, 224), device=device)
#     else:
#         if lm_name == '2d106det':
#             # not necessary to test
#             return

#         model = Model((112, 112), device=device)

#     imgs, n_exp = img_params
#     if not isinstance(imgs, list):
#         imgs = [imgs]

#     imgs_path = [Path(__file__).parent / f'test_data/{img}.jpg'
#                  for img in imgs]

#     out_crop = model(imgs_path)
#     template = getattr(model, 'template', None)
#     if Model == LandmarkBboxCropModel:
#         f_out = Path(__file__).parent / f'test_viz/crop/{str(model)}_lm-{lm_name}_exp-{n_exp}.jpg'
#     else:
#         f_out = Path(__file__).parent / f'test_viz/crop/{str(model)}_exp-{n_exp}.jpg'

#     out_crop.visualize(imgs_path, template=template, f_out=f_out)


videos = [
    ('example_vid.mp4', 1),  # easy
#    ('one_face.mp4', 1),    # also moves out of frame
#    ('three_faces.mp4', 3),  # three faces, incl sunglasses
#    ('four_faces.mp4', 4)    # occlusion, moving in and out of frame
]

@pytest.mark.parametrize('Model', [LandmarkBboxCropModel])#, LandmarkAlignCropModel])
@pytest.mark.parametrize('video', videos)
def test_crop_model_vid(Model, video):
    if Model == LandmarkBboxCropModel:
        model = Model('2d106det', (224, 224))
    else:
        model = Model((448, 448))

    video, n_exp = video
    video_path = Path(__file__).parent / f'test_data/{video}'
    loader = VideoLoader(video_path, batch_size=32, loglevel='WARNING')

    crops, imgs = [], []
    for batch in loader:
        out_crop = model(batch)
        crops.append(out_crop)
        imgs.append(batch)

    crops = CropResults.from_batches(crops)
    crops.sort(dist_threshold=250, present_threshold=0.1)
    imgs = torch.concatenate(imgs)
    f_out = Path(__file__).parent / f'test_viz/crop/{str(model)}_{video}'

    template = getattr(model, 'template', None)
    crops.visualize(imgs, f_out=f_out, template=template, video=True, show_crop=True)

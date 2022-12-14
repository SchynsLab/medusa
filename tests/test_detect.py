from pathlib import Path

import pytest
import torch
from test_utils import _check_gha_compatible

from medusa.detect import SCRFDetector, YunetDetector
from medusa.io import VideoLoader
from medusa.containers.results import BatchResults


imgs = ['no_face', 'one_face', 'two_faces', 'three_faces',
        ['no_face', 'one_face'], ['no_face', 'two_faces'],
        ['one_face', 'two_faces', 'three_faces']]
n_exp = [0, 1, 2, 3, 1, 2, 6]
img_params = zip(imgs, n_exp)


@pytest.mark.parametrize('Detector', [SCRFDetector, YunetDetector])
@pytest.mark.parametrize('imgs,n_exp', img_params)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_detector_imgs(Detector, imgs, n_exp, device):
    if not _check_gha_compatible(device):
        return

    model = Detector(device=device)

    if not isinstance(imgs, list):
        imgs = [imgs]

    imgs_path = [Path(__file__).parent / f'test_data/{img}.jpg'
                 for img in imgs]
    out_det = model(imgs_path)
    out_det = BatchResults(device=device, **out_det)

    n_det = len(getattr(out_det, 'conf', []))
    assert(n_det == n_exp)

    for key in ['conf', 'bbox', 'lms']:
        if n_det == 0:
            assert(not hasattr(out_det, key))
        else:
            assert(getattr(out_det, key).shape[0] == n_det)

    f_out = Path(__file__).parent / f'test_viz/detection/{str(model)}_exp-{n_exp}.jpg'
    out_det.visualize(f_out, imgs_path)


videos = [
    ('example_vid.mp4', 1),  # easy
    ('one_face.mp4', 1),    # also moves out of frame
    ('three_faces.mp4', 3),  # three faces, incl sunglasses
    ('four_faces.mp4', 4)    # occlusion, moving in and out of frame
]

@pytest.mark.parametrize('Detector', [SCRFDetector, YunetDetector])
@pytest.mark.parametrize('video,n_exp', videos)
def test_detector_vid(Detector, video, n_exp):
    video_path = Path(__file__).parent / f'test_data/{video}'
    loader = VideoLoader(video_path, batch_size=8)
    model = Detector()

    results = BatchResults()
    for batch in loader:
        out_det = model(batch)
        results.add(imgs=batch, **out_det)

    results.concat()
    assert(torch.all(torch.diff(results.img_idx) >= 0))
    results.sort_faces('lms')

    f_out = Path(__file__).parent / f'test_viz/detection/{str(model)}_{video}'
    results.visualize(f_out, results.imgs, video=True)

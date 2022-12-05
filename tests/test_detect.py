from pathlib import Path

import pytest
import torch
from test_utils import _check_gha_compatible

from medusa.detect import RetinanetDetector, YunetDetector
from medusa.detect.base import DetectionResults
from medusa.io import VideoLoader

imgs = ['no_face', 'one_face', 'two_faces', 'three_faces',
        ['no_face', 'one_face'], ['no_face', 'two_faces'],
        ['one_face', 'two_faces', 'three_faces']]
n_exp = [0, 1, 2, 3, 1, 2, 6]
img_params = zip(imgs, n_exp)


@pytest.mark.parametrize('Detector', [RetinanetDetector, YunetDetector])
@pytest.mark.parametrize('img_params', img_params)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_detector_imgs(Detector, img_params, device):
    if not _check_gha_compatible(device):
        return

    model = Detector(device=device)
    imgs, n_exp = img_params

    if not isinstance(imgs, list):
        imgs = [imgs]

    imgs_path = [Path(__file__).parent / f'test_data/{img}.jpg'
                 for img in imgs]
    out_det = model(imgs_path)

    n_det = len(out_det)
    assert(n_det == n_exp)

    for key in ['conf', 'bbox', 'lms']:
        if n_det == 0:
            assert(getattr(out_det, key) is None)
        else:
            assert(getattr(out_det, key).shape[0] == n_det)

    f_out = Path(__file__).parent / f'test_viz/detection/{str(model)}_exp-{n_exp}.jpg'
    out_det.visualize(imgs_path, f_out=f_out)


videos = [
    ('example_vid.mp4', 1),  # easy
    ('one_face.mp4', 1),    # also moves out of frame
    ('three_faces.mp4', 3),  # three faces, incl sunglasses
    ('four_faces.mp4', 4)    # occlusion, moving in and out of frame
]

@pytest.mark.parametrize('Detector', [RetinanetDetector, YunetDetector])
@pytest.mark.parametrize('video', videos)
def test_detector_vid(Detector, video):
    video, n_exp = video
    video_path = Path(__file__).parent / f'test_data/{video}'
    loader = VideoLoader(video_path)
    model = Detector()

    detections, imgs = [], []
    for batch in loader:
        detections.append(model(batch))
        imgs.append(batch)

    detections = DetectionResults.from_batches(detections)
    assert(torch.all(torch.diff(detections.img_idx) >= 0))

    detections.sort(dist_threshold=200, present_threshold=0.1)
    #assert(len(detections.face_idx.unique()) == n_exp)

    imgs = torch.concatenate(imgs)
    f_out = Path(__file__).parent / f'test_viz/detection/{str(model)}_{video}'
    detections.visualize(imgs, f_out=f_out, video=True)

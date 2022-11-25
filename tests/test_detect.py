import os
import torch
import pytest
from pathlib import Path
from medusa.detect import YunetDetector, RetinanetDetector

from test_utils import _check_gha_compatible

imgs = ['no_face.jpg', 'one_face.jpg', 'two_faces.jpg', 'three_faces.jpg']

@pytest.mark.parametrize('Detector', [RetinanetDetector, YunetDetector])
@pytest.mark.parametrize('img_params', zip(imgs, [None, 1, 2, 3]))
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_detector(Detector, img_params, batch_size, device):

    if not _check_gha_compatible(device):
        return

    model = Detector(device=device)
    img, exp_n_face = img_params
    img_path = Path(__file__).parent / f'test_data/detection/{img}'
    img_path = batch_size * [img_path]
    out = model(img_path)

    idx = out['idx']
    n_detections = idx[~torch.isnan(idx)].numel()

    if exp_n_face is None:
        for value in out.values():
            assert(torch.all(torch.isnan(value)))
    else:
        for key, value in out.items():
            assert(n_detections == exp_n_face * batch_size)
            assert(value.shape[0] == n_detections)

    f_out = Path(__file__).parent / f'test_viz/detection/{str(model)}_{img}'
    model.visualize(img_path, out, f_out=f_out)

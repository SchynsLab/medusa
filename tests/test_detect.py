import os
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
    conf, bbox, lms = model(img_path)

    assert(len(conf) == len(bbox) == len(lms) == batch_size)

    if exp_n_face is None:
        for output in (conf, bbox, lms):
            assert(all(outp is None for outp in output))
        
    else:
        assert(conf[0].shape[0] == bbox[0].shape[0] == lms[0].shape[0])            
        n_detected = conf[0].shape[0]
        assert(n_detected == exp_n_face)

    if batch_size == 1:
        f_out = Path(__file__).parent / f'test_viz/detection/{str(model)}_{img}'
        model.visualize(img_path, bbox, lms, conf, f_out=f_out)

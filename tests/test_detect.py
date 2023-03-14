import os
from pathlib import Path

import pytest
import torch
from conftest import _is_gha_compatible

from medusa.containers.results import BatchResults
from medusa.detect import SCRFDetector, YunetDetector
from medusa.io import VideoLoader
from medusa.data import get_example_frame


@pytest.mark.parametrize("Detector", [SCRFDetector, YunetDetector])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_detector_device(Detector, device):
    if not _is_gha_compatible(device):
        return

    img = get_example_frame()
    model = Detector(device=device)
    out_det = model(img)
    assert(out_det['lms'].device.type == device)


@pytest.mark.parametrize('det_size', [(224, 224), (640, 640)])
def test_scrfd_det_size(det_size):
    img = get_example_frame()
    model = SCRFDetector(det_size=det_size)
    _ = model(img)


@pytest.mark.parametrize("Detector", [SCRFDetector, YunetDetector])
@pytest.mark.parametrize(
    "imgs_test", [0, 1, 2, 3, 4, [0, 1], [1, 2], [1, 2, 3, 4]], indirect=True
)
def test_detector_imgs(Detector, imgs_test):
    model = Detector()
    imgs, n_exp = imgs_test
    out_det = model(imgs)
    out_det = BatchResults(**out_det)

    n_det = len(getattr(out_det, "conf", []))
    assert n_det == n_exp

    for key in ["conf", "bbox", "lms"]:
        if n_det == 0:
            assert not hasattr(out_det, key)
        else:
            assert getattr(out_det, key).shape[0] == n_det

    f_out = Path(__file__).parent / f"test_viz/detection/{str(model)}_exp-{n_exp}.jpg"
    out_det.visualize(f_out, imgs)


@pytest.mark.parametrize("Detector", [SCRFDetector, YunetDetector])
@pytest.mark.parametrize("video_test", [0, 1, 2, 3, 4], indirect=True)
def test_detector_vid(Detector, video_test):
    loader = VideoLoader(video_test, batch_size=8)
    model = Detector()

    results = BatchResults()
    for batch in loader:
        out_det = model(batch)
        results.add(imgs=batch, **out_det)

    results.concat()
    if getattr(results, "img_idx", None) is None:
        return

    assert torch.all(torch.diff(results.img_idx) >= 0)
    results.sort_faces("lms")

    if 'GITHUB_ACTIONS' in os.environ:
        return

    f_out = (
        Path(__file__).parent / f"test_viz/detection/{str(model)}_{video_test.stem}.mp4"
    )
    results.visualize(f_out, results.imgs, video=True)

import os
from pathlib import Path

import pytest
import torch
from conftest import _is_device_compatible

from medusa.containers.results import BatchResults
from medusa.detect import SCRFDetector, YunetDetector
from medusa.io import VideoLoader
from medusa.data import get_example_image, get_example_video
from medusa.defaults import DEVICE


@pytest.mark.parametrize("Detector", [SCRFDetector, YunetDetector])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_detector_device(Detector, device):

    if not _is_device_compatible(device):
        return

    img = get_example_image(device=device)
    model = Detector(device=device)
    model.to(device)
    out_det = model(img)
    assert(out_det['lms'].device.type == device)


@pytest.mark.parametrize('det_size', [(224, 224), (640, 640)])
def test_scrfd_det_size(det_size):
    img = get_example_image()
    model = SCRFDetector(det_size=det_size)
    _ = model(img)


@pytest.mark.parametrize("Detector", [SCRFDetector, YunetDetector])
@pytest.mark.parametrize("n_faces", [0, 1, 2, 3, 4, [0, 1], [1, 2], [1, 2, 3, 4]])
def test_detector_imgs(Detector, n_faces):

    imgs = get_example_image(n_faces)
    if isinstance(n_faces, int):
        n_exp = n_faces
    else:
        n_exp = sum(n_faces)

    model = Detector()

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
@pytest.mark.parametrize("n_faces", [0, 1, 2, 3, 4])
def test_detector_vid(Detector, n_faces):

    video_test = get_example_video(n_faces)
    loader = VideoLoader(video_test, batch_size=8)
    model = Detector()

    results = BatchResults()
    for batch in loader:
        batch = batch.to(DEVICE)
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

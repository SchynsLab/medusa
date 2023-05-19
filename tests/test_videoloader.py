import pytest
from conftest import _is_gha_compatible

from medusa.data import get_example_video
from medusa.io import VideoLoader


def test_videoloader_full_iteration():
    vid = get_example_video(return_videoloader=False)
    loader = VideoLoader(vid, device="cpu")

    n_expected = len(loader.dataset)
    n_actual = 0
    for img_batch in loader:
        n_actual += img_batch.shape[0]

    assert n_actual == n_expected


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_videoloader_params(batch_size, device):
    if not _is_gha_compatible(device):
        return

    vid = get_example_video(return_videoloader=False)
    loader = VideoLoader(vid, device=device, batch_size=batch_size)

    metadata = loader.get_metadata()
    img_batch = next(iter(loader))

    assert img_batch.shape[0] == batch_size
    assert img_batch.shape[2:] == metadata["img_size"][::-1]


@pytest.mark.parametrize("frames", [[0, 10], [0, 1, 2, 3], [10, 20, 30, 40, 50]])
def test_videoloader_subset(frames):

    vid = get_example_video()
    loader = VideoLoader(vid, dataset_type='subset', frames=frames, batch_size=2)
    i = 0
    for batch in loader:
        i += batch.shape[0]

    assert(i == len(frames))

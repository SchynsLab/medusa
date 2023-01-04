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
    assert img_batch.device.type == device
    assert img_batch.shape[1:3] == metadata["img_size"][::-1]

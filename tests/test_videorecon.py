import tempfile

import pytest

from medusa.data import get_example_video
from medusa.recon import videorecon


@pytest.mark.parametrize("n_frames", [5, None])
def test_videorecon(n_frames):
    """Tests the videorecon Python interface."""

    vid = get_example_video(as_path=False)
    data = videorecon(vid, recon_model='mediapipe', device='cpu', n_frames=n_frames)

    with tempfile.NamedTemporaryFile() as f_out:
        data.save(f_out.name)

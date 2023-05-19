import os
from pathlib import Path

import pytest

from medusa.defaults import DEVICE
from medusa.data import get_example_data4d, get_example_video
from medusa.render import VideoRenderer


@pytest.mark.parametrize("n_faces", [1, 4])
def test_render_video(n_faces, device=DEVICE):
    if 'GITHUB_ACTIONS' in os.environ:
        # Too slow for Github Actions
        return

    vid = get_example_video(n_faces)
    data = get_example_data4d(n_faces, load=True, model='emoca-coarse', device=device)
    data.apply_vertex_mask('face')
    f_out = (
        Path(__file__).parent
        / f"test_viz/render/f{n_faces}_face.mp4"
    )
    renderer = VideoRenderer(background=vid)
    renderer.render(f_out, data)


@pytest.mark.parametrize("background", [(0, 0, 0), (255, 255, 255)])
def test_render_video_background(background):
    if 'GITHUB_ACTIONS' in os.environ:
        # Too slow for Github Actions
        return

    vid = get_example_video()
    data = get_example_data4d(load=True, model='emoca-coarse')
    data.apply_vertex_mask('face')
    f_out = (
        Path(__file__).parent
        / f"test_viz/render/example_vid_background-{background}.mp4"
    )
    renderer = VideoRenderer(background=background)
    renderer.render(f_out, data)

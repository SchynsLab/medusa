import pytest
from pathlib import Path
from medusa.data import get_example_data4d, get_example_video
from medusa.preproc import bw_filter
from medusa.render import VideoRenderer

@pytest.mark.parametrize('low_pass', [10, 2])
@pytest.mark.parametrize('high_pass', [0.0001, 0.005])
def test_filter(low_pass, high_pass):
    data = get_example_data4d(load=True)
    fps = data.video_metadata['fps']
    data.v = bw_filter(data.v, fps, low_pass, high_pass)
    data.mat = bw_filter(data.mat, fps, low_pass, high_pass)
    f_out = Path(__file__).parent / f'test_viz/preproc/test_filt_lp-{low_pass}_hp-{high_pass}.mp4'
    renderer = VideoRenderer()
    renderer.render(f_out, data, video=get_example_video())

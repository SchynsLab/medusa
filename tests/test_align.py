import pytest
from pathlib import Path
from medusa.preproc.align import estimate_alignment
from medusa.data import get_example_h5


@pytest.mark.parametrize('model,topo', [('mediapipe', 'mediapipe'), ('emoca-coarse', 'flame-coarse')])
def test_estimate_alignment(model, topo):

    data = get_example_h5(load=True, model=model)
    data.mat = estimate_alignment(data.v, topo, estimate_scale=True)
    data.to_local()
    f_out = Path(__file__).parent / f"test_viz/preproc/test_align_estimate_model-{model}.mp4"
    data.render_video(f_out)

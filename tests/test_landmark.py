from pathlib import Path
import pytest

from medusa.landmark import RetinafaceLandmarkModel
from medusa.data import get_example_image
from medusa.containers import BatchResults


@pytest.mark.parametrize("lm_model_name", ["2d106det", "1k3d68"])
@pytest.mark.parametrize("n_faces", [0, 1, 2, 3, 4, [0, 1], [0, 1, 2], [0, 1, 2, 3, 4]])
def test_retinaface_landmark_model(lm_model_name, n_faces):

    imgs = get_example_image(n_faces)
    model = RetinafaceLandmarkModel(lm_model_name)
    out_lms = model(imgs)

    if isinstance(n_faces, int):
        n_exp = n_faces
    else:
        n_exp = sum(n_faces)

    if n_exp == 0:
        assert out_lms['lms'] is None
    else:
        assert out_lms['lms'].shape[0] == n_exp

    out_lms = BatchResults(**out_lms)
    f_out = Path(__file__).parent / f"test_viz/landmark/{str(lm_model_name)}_exp-{n_exp}.jpg"
    out_lms.visualize(f_out, imgs)

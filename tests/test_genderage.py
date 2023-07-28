from pathlib import Path
import pytest

from medusa.recognize import RetinafaceGenderAgeModel
from medusa.data import get_example_image
from medusa.containers import BatchResults
from medusa.data import get_external_data_config


isf_path = get_external_data_config('insightface_path').parent


@pytest.mark.parametrize("n_faces", [1, 2, 3, 4])
@pytest.mark.parametrize("isf_model_name", ["buffalo_l", "antelopev2"])
def test_retinaface_landmark_model(n_faces, isf_model_name):

    model_path = isf_path / isf_model_name / 'genderage.onnx'

    imgs = get_example_image(n_faces)
    model = RetinafaceGenderAgeModel(model_path=model_path)
    out = model(imgs)

    for key in ['age', 'gender']:
        assert(out[key].shape[0] == n_faces)

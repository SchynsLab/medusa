import torch
import pytest
from medusa.recognize import RetinafaceRecognitionModel
from medusa.data import get_example_image
from medusa.data import get_external_data_config


isf_path = get_external_data_config('insightface_path').parent

@pytest.mark.parametrize('model_name', ['buffalo_l', 'antelopev2'])
@pytest.mark.parametrize("n_faces", [1, 2, 3, 4])
def test_retinaface_recognition_model(model_name, n_faces):

    if model_name == 'buffalo_l':
        model_path = isf_path / model_name / 'w600k_r50.onnx'
    else:
        model_path = isf_path / model_name / 'glintr100.onnx'

    if not model_path.is_file():
        return

    imgs = get_example_image(n_faces)
    model = RetinafaceRecognitionModel(model_path=model_path)
    out_emb = model(imgs)

    assert(torch.is_tensor(out_emb))
    assert(out_emb.shape == (n_faces, 512))

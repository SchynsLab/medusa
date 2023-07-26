import torch
import pytest

from medusa.recognize import RetinafaceRecognitionModel
from medusa.data import get_example_image


@pytest.mark.parametrize("n_faces", [1, 2, 3, 4])
def test_retinaface_recognition_model(n_faces):

    imgs = get_example_image(n_faces)
    model = RetinafaceRecognitionModel()
    out_emb = model(imgs)

    assert(torch.is_tensor(out_emb))
    assert(out_emb.shape == (n_faces, 512))

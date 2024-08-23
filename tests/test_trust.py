import torch
import pytest
from pathlib import Path
from torchvision.utils import save_image
from medusa.albedo import TRUST
from medusa.crop import RandomSquareCropModel, BboxCropModel
from medusa.data import get_example_image


@pytest.mark.parametrize("n_faces", [1, 2, 3, 4])
def test_trust(n_faces):

    face_crop_model = BboxCropModel(output_size=(224, 224), scale=1.6, scale_orig=1)
    scene_crop_model = RandomSquareCropModel(output_size=(224, 224))
    
    img = get_example_image(n_faces=n_faces)
    face_crop = face_crop_model(img)['imgs_crop']
    scene_crop = scene_crop_model(img)['imgs_crop']
    scene_crop = scene_crop.repeat(face_crop.shape[0], 1, 1, 1)

    model = TRUST()
    with torch.inference_mode():
        out = model(face_crop, scene_crop)
    
    dir_out = Path(__file__).parent / "test_viz/albedo"
    save_image(out['albedo'].float(), dir_out / f'albedo_n-{n_faces}.png', normalize=True)    
    save_image(face_crop.float(), dir_out / f'crop-face_n-{n_faces}.png', normalize=True)
    save_image(scene_crop.float(), dir_out / f'crop-scene_n-{n_faces}.png', normalize=True)

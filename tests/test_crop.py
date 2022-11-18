import os
import cv2
import pytest
from pathlib import Path
from medusa.data import get_example_frame
from medusa.crop import FanCropModel, InsightfaceCropModel


@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_fan_crop(device):

    if 'GITHUB_ACTIONS' in os.environ and device == 'cuda':
        return

    img = get_example_frame(load_numpy=True)
    crop_model = FanCropModel(device=device, return_bbox=True)
    img_crop, crop_mat, bbox = crop_model(img)
    
    assert(img_crop.shape == (1, 3, 224, 224))
    assert(crop_mat.shape == (1, 3, 3))
    assert(bbox.shape == (1, 4, 2))
    
    img_crop = crop_model.to_numpy(img_crop)
    f_out = Path(__file__).parent / 'test_viz/fan_crop_img.png'
    cv2.imwrite(str(f_out), img_crop.squeeze())

    f_out = Path(__file__).parent / 'test_viz/fan_crop_bbox.png'
    crop_model.visualize_bbox(img, bbox, f_out)


def test_insightface_crop():

    img = get_example_frame(load_numpy=True)
    crop_model = InsightfaceCropModel(device='cpu', return_bbox=True)
    img_crop, crop_mat, bbox = crop_model(img)    
    
    assert(img_crop.shape == (1, 3, 112, 112))
    assert(bbox.shape == (1, 4, 2))

    img_crop = crop_model.to_numpy(img_crop)
    f_out = Path(__file__).parent / 'test_viz/insightface_crop_img.png'
    cv2.imwrite(str(f_out), img_crop.squeeze())

    f_out = Path(__file__).parent / 'test_viz/insightface_crop_bbox.png'
    crop_model.visualize_bbox(img, bbox, f_out)


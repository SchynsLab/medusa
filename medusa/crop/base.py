import torch
from pathlib import Path
from torchvision.utils import draw_keypoints, save_image, draw_bounding_boxes
from kornia.geometry.linalg import transform_points

from .. import DEVICE
from ..io import load_inputs


class BaseCropModel:
    pass


class CropResults:

    def __init__(self, imgs_crop, crop_mats, lms, idx, device=DEVICE):
        self.imgs_crop = imgs_crop
        self.crop_mats = crop_mats
        self.lms = lms
        self.idx = idx
        self.device = device

    def visualize(self, imgs, f_out, **kwargs):

        if not isinstance(f_out, Path):
            f_out = Path(f_out)

        # batch_size x h x w x 3
        imgs = load_inputs(imgs, load_as='torch', channels_first=True, device=self.device, dtype='uint8')
        n_images = imgs.shape[0]

        for i_img in range(n_images):
            # skip if there are no detections at all 
            if self.idx is None:
                continue
            # or only the current image has no detections
            elif i_img not in self.idx:
                continue

            img = imgs[i_img, ...]
            idx = self.idx == i_img
            b, c, h, w = self.imgs_crop[idx].shape

            # Create the bbox used for cropping by warping top-left and bottom-right corner
            # to the original image space using the previously estimated crop matrix
            corners = torch.tensor([[0, 0], [h-1, w-1]], dtype=torch.float32, device=self.device)
            bbox_crop = corners.repeat(b, 1, 1)
            crop_mats = torch.inverse(self.crop_mats[idx])
            bbox = transform_points(crop_mats, bbox_crop).reshape((b, 4))
            img = draw_bounding_boxes(img, bbox, labels=None, colors=(255, 0, 0))

            # Also transform landmark points back to the original image space
            # (maybe don't warp them to cropped image space at all?)
            lms = transform_points(crop_mats, self.lms[idx])           
            img = draw_keypoints(img, lms, colors=(0, 255, 0), radius=1)
            imgs[i_img] = img
        
        if f_out.is_file():
            # Remove if exists already
            f_out.unlink()

        save_image(imgs.float(), fp=f_out, normalize=True, **kwargs)

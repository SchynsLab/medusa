from .base import BaseCropModel
from ..defaults import DEVICE

import torch
from torchvision.transforms import Resize
import numpy as np


class RandomSquareCropModel(BaseCropModel):
    # Not sure about 224x224 for TRUST
    def __init__(self, output_size=(224, 224), device=DEVICE):
        super().__init__()
        self.device = device
        self.to(device).eval()

        if output_size[0] != output_size[1]:
            raise ValueError("Output size should be square!")

        self.resizer = Resize(size=output_size)

    def __str__(self):
        return "randomsquarecrop"

    def forward(self, imgs):
        """Aligns and crops images to the desired size.

        Parameters
        ----------
        imgs : str, Path, tuple, list, array_like, torch.tensor
            A path to an image, or a tuple/list of them, or already loaded images
            as a torch.tensor or numpy array

        Returns
        -------
        out_crop : dict
            Dictionary with cropping outputs; includes the keys "imgs_crop" (cropped
            images) and "crop_mat" (3x3 crop matrices)
        """
        # Load images here instead of in detector to avoid loading them twice
        
        out = []
        for i in range(imgs.shape[0]):
            image = imgs[i, ...]
            scene_h, scene_w = image.shape[:2]
            if scene_w > scene_h:
                sq_size = scene_h
                random_left = np.random.randint(scene_w - sq_size)
                square_scene = image[0:sq_size, random_left:random_left + sq_size]
            elif scene_h > scene_w:
                sq_size = scene_w
                random_top = np.random.randint(scene_h - sq_size)
                square_scene = image[random_top: random_top+sq_size, 0:sq_size]
            else:
                square_scene = image.clone()

            square_scene = self.resizer(square_scene)
            out.append(square_scene)

        square_scene = torch.stack(out, dim=0)
        return square_scene
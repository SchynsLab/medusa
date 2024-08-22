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
        
        out = []
        for i in range(imgs.shape[0]):
            img = imgs[i, ...]
            scene_h, scene_w = img.shape[1:]

            if scene_w > scene_h:
                sq_size = scene_h
                random_left = np.random.randint(scene_w - sq_size)
                scene = img[..., :sq_size, random_left:random_left + sq_size]
            elif scene_h > scene_w:
                sq_size = scene_w
                random_top = np.random.randint(scene_h - sq_size)
                scene = img[..., random_top: random_top+sq_size, :sq_size]
            else:
                scene = img.clone()

            scene = self.resizer(scene)
            out.append(scene)

        scene = torch.stack(out, dim=0)
        return {'imgs_crop': scene}
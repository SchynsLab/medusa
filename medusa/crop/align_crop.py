"""Module with an implementation of a "crop model" that aligns an image to a
template based on a set of landmarks (based on an implementation from
Insightface)."""

import numpy as np
import torch
from kornia.geometry.transform import warp_affine

from .base import BaseCropModel
from ..io import load_inputs
from ..defaults import DEVICE
from ..detect import SCRFDetector
from ..transforms import estimate_similarity_transform


TEMPLATE = torch.Tensor(
    np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
)
"""The 5-landmark template used by Insightface (e.g. in their arcface implementation).
The coordinates are relative to an image of size 112 x 112."""


class LandmarkAlignCropModel(BaseCropModel):
    """Cropping model based on functionality from the ``insightface`` package,
    as used by MICA (https://github.com/Zielon/MICA).

    Parameters
    ----------
    name : str
        Name of underlying insightface model
    det_size : tuple
        Image size for detection
    target_size : tuple
        Length 2 tuple with desired width/heigth of cropped image; should be (112, 112)
        for MICA
    det_thresh : float
        Detection threshold (higher = more stringent)
    device : str
        Either 'cuda' (GPU) or 'cpu'

    Examples
    --------
    To crop an image to be used for MICA reconstruction:

    >>> from medusa.data import get_example_frame
    >>> crop_model = LandmarkAlignCropModel()
    >>> img = get_example_frame()  # path to jpg image
    >>> out = crop_model(img)
    """

    def __init__(
        self,
        output_size=(112, 112),
        template=TEMPLATE,
        detector=SCRFDetector,
        device=DEVICE,
        **kwargs
    ):

        self.output_size = output_size  # h, w
        self.template = template * (output_size[0] / 112.0)
        self._det_model = detector(device=device, **kwargs)
        self.device = device

        if output_size[0] != output_size[1]:
            raise ValueError("Output size should be square!")

    def __str__(self):
        return "aligncrop"

    def __call__(self, imgs):
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
            images) and "crop_mats" (3x3 crop matrices)
        """
        # Load images here instead of in detector to avoid loading them twice
        imgs = load_inputs(
            imgs, load_as="torch", channels_first=True, device=self.device
        )
        b, c, h, w = imgs.shape
        out_det = self._det_model(imgs)

        if out_det.get("conf", None) is None:
            return {"imgs_crop": None, "crop_mats": None, **out_det}

        # Estimate transform landmarks -> template landmarks
        crop_mats = estimate_similarity_transform(
            out_det["lms"], self.template, estimate_scale=True
        )
        imgs_stacked = imgs[out_det["img_idx"]]
        imgs_crop = warp_affine(
            imgs_stacked, crop_mats[:, :2, :], dsize=self.output_size
        )

        out_crop = {"imgs_crop": imgs_crop, "crop_mats": crop_mats, **out_det}

        return out_crop

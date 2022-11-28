""" Crop model adapted from the insightface implementation. By reimplementing
it here, insightface does not have to be installed.

Please see the LICENSE file in the current directory for the license that
is applicable to this implementation.
"""

import torch
import numpy as np
from kornia.geometry.transform import warp_affine
from kornia.geometry.linalg import transform_points

from .. import DEVICE
from ..io import load_inputs
from .base import BaseCropModel, CropResults
from ..detect import RetinanetDetector
from ..transforms import estimate_similarity_transform


# Arcface template as defined by Insightface
TEMPLATE = torch.Tensor(np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32
))


class LandmarkAlignCropModel(BaseCropModel):
    """ Cropping model based on functionality from the ``insightface`` package, as used
    by MICA (https://github.com/Zielon/MICA).

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
    >>> crop_model = InsightfaceCropModel(device='cpu')
    >>> img = get_example_frame()  # path to jpg image
    >>> crop_img = crop_model(img)
    >>> crop_img.shape
    torch.Size([1, 3, 112, 112])
    """    

    def __init__(self, output_size=(112, 112), template=TEMPLATE, detector=RetinanetDetector,
                 return_lmk=False, device=DEVICE, **kwargs):

        self.output_size = output_size  # h, w
        self.template = template * (output_size[0] / 112.)
        self._det_model = detector(device=device, **kwargs)
        self.return_lmk = return_lmk
        self.device = device

        if output_size[0] != output_size[1]:
            raise ValueError("Output size should be square!")

    def __str__(self):
        return 'aligncrop'

    def __call__(self, imgs):
        
        # Load images here instead of in detector to avoid loading them twice
        imgs = load_inputs(imgs, load_as='torch', channels_first=True, device=self.device)
        out_det = self._det_model(imgs)

        if len(out_det) == 0:
            out_crop = CropResults(None, None, None, None, self.device)
            return out_crop

        # Estimate transform landmarks -> template landmarks
        crop_mats = estimate_similarity_transform(out_det.lms, self.template, estimate_scale=True)
        imgs_stacked = imgs[out_det.idx]
        imgs_crop = warp_affine(imgs_stacked, crop_mats[:, :2, :], dsize=self.output_size)
        lms = transform_points(crop_mats, out_det.lms)
        out_crop = CropResults(imgs_crop, crop_mats, lms, out_det.idx, device=self.device)    

        return out_crop

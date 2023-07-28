"""Module with a "crop model" which crops an image by creating a bounding box
based on a set of existing (2D) landmarks.

Based on the implementation in DECA (see
../recon/flame/deca/license.md).
"""

import torch
from kornia.geometry.transform import warp_affine

from ..defaults import DEVICE
from ..detect import SCRFDetector
from ..transforms import estimate_similarity_transform
from .base import BaseCropModel


class BboxCropModel(BaseCropModel):
    """A model that crops an image by creating a bounding box based on a set of
    face landmarks; based on the implementation from DECA.

    Parameters
    -----------
    name : str
        Name of the landmark model from Insightface that should be used; options are
        '2d106det' (106 landmarks) or '1k3d68' (68 landmarks)
    output_size : tuple[int]
        Desired size of the cropped image
    detector : BaseDetector
        A Medusa-based detector
    device : str
        Either 'cuda' (GPU) or 'cpu'
    """

    def __init__(self, lms_model_name="2d106det", output_size=(224, 224),
                 device=DEVICE):

        # Avoids circular import
        from ..landmark import RetinafaceLandmarkModel

        # alternative: 1k3d68, 2d106det
        super().__init__()
        self.output_size = output_size  # h, w
        self.device = device
        self._lms_model = RetinafaceLandmarkModel(lms_model_name, device=device)
        self.to(device).eval()

    def __str__(self):
        return "bboxcrop"

    def _create_bbox(self, lm, scale=1.25):
        """Creates a bounding box (bbox) based on the landmarks by creating a
        box around the outermost landmarks (+10%), as done in the original DECA
        model.

        Parameters
        ----------
        lm : torch.tensor
            Float tensor with landmarks of shape L (landmarks) x 2
        scale : float
            Factor to scale the bounding box with
        """
        left = torch.min(lm[:, :, 0], dim=1)[0]
        right = torch.max(lm[:, :, 0], dim=1)[0]
        top = torch.min(lm[:, :, 1], dim=1)[0]
        bottom = torch.max(lm[:, :, 1], dim=1)[0]

        # scale and 1.1 are DECA constants
        orig_size = (right - left + bottom - top) / 2 * 1.1
        size = orig_size * scale  # to int?

        # b x 2 (center coords)
        center = torch.stack(
            [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0], dim=1
        )

        b = lm.shape[0]
        bbox = torch.zeros((b, 4, 2), device=self.device)
        bbox[:, 0, :] = center - size[:, None] / 2
        bbox[:, 1, 0] = center[:, 0] - size / 2
        bbox[:, 1, 1] = center[:, 1] + size / 2
        bbox[:, 2, 0] = center[:, 0] + size / 2
        bbox[:, 2, 1] = center[:, 1] - size / 2
        bbox[:, 3, :] = center + size[:, None] / 2

        return bbox

    def forward(self, imgs):
        """Crops images to the desired size.

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
        out_lms = self._lms_model(imgs)
        lms = out_lms['lms']

        if lms is None:
            return {**out_lms, "imgs_crop": None, "crop_mat": None}

        bbox = self._create_bbox(lms)

        # Estimate a transform from the new bbox to the final
        # cropped image space (probably 224 x 224 for DECA-based models)
        w_out, h_out = self.output_size
        dst = torch.tensor(
            [[0, 0], [0, w_out - 1], [h_out - 1, 0]],
            dtype=torch.float32,
            device=self.device,
        )
        dst = dst.repeat(lms.shape[0], 1, 1)
        crop_mat = estimate_similarity_transform(
            bbox[:, :3, :], dst, estimate_scale=True
        )

        # Convert to torchvision format (xmin, ymin, xmax, ymax)
        bbox = bbox[:, [0, 3], :].reshape(-1, 4)

        # Finally, warp the original images (uncropped) images to the final
        # cropped space
        imgs_stack = imgs[out_lms["img_idx"]]
        imgs_crop = warp_affine(imgs_stack, crop_mat[:, :2, :], dsize=(h_out, w_out))
        out_crop = {
            **out_lms,
            "bbox": bbox,
            "imgs_crop": imgs_crop.to(dtype=torch.uint8),
            "crop_mat": crop_mat,
        }

        return out_crop


class InsightfaceBboxCropModel(BaseCropModel):
    """Crop model based on a (scaled and) square bounding box as implemented by
    insightface.

    Parameters
    ----------
    device : str
        Either 'cuda' (GPU) or 'cpu'
    """
    def __init__(self, output_size=(192, 192), scale_factor=1.5, device=DEVICE):
        super().__init__()
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.device = device
        self._det_model = SCRFDetector(device=device)
        self.to(device).eval()

    def forward(self, imgs):
        """Runs the crop model on a set of images.

        Parameters
        ----------
        imgs : torch.tensor
            A torch.tensor of shape N (batch) x C x H x W

        Returns
        -------
        out_crop : dict
            Dictionary with cropping outputs; includes the keys "imgs_crop" (cropped
            images) and "crop_mat" (3x3 crop matrices)
        """

        out_det = self._det_model(imgs)

        if out_det.get("conf", None) is None:
            return {**out_det, "imgs_crop": None, "crop_mat": None}

        n_det = out_det["bbox"].shape[0]
        bbox = out_det["bbox"]
        imgs_stack = imgs[out_det["img_idx"]]

        bw, bh = (bbox[:, 2] - bbox[:, 0]), (bbox[:, 3] - bbox[:, 1])
        center = torch.stack(
            [(bbox[:, 2] + bbox[:, 0]) / 2, (bbox[:, 3] + bbox[:, 1]) / 2], dim=1
        )

        oh, ow = self.output_size
        scale = oh / (torch.maximum(bw, bh) * self.scale_factor)

        # Construct initial crop matrix based on rough bounding box,
        # then crop images to size 192 x 192
        M = torch.eye(3, device=self.device).repeat(n_det, 1, 1) * scale[:, None, None]
        M[:, 2, 2] = 1  # should always be 1
        M[:, :2, 2] = -1 * center * scale[:, None] + oh / 2
        imgs_crop = warp_affine(imgs_stack, M[:, :2, :], dsize=self.output_size)

        out_crop = {
            **out_det,
            "imgs_crop": imgs_crop.to(dtype=torch.uint8),
            "crop_mat": M,
        }

        return out_crop

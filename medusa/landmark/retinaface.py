"""A landmark detection model based on Insightface's Retinaface model, but implemented
in PyTorch (but parts of it run with ONNX model), so can be fully run on GPU (no numpy
necessary)."""

import torch
from torch import nn

from ..onnx import OnnxModel
from ..detect import SCRFDetector
from ..defaults import DEVICE

from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import warp_affine


class RetinafaceLandmarkModel(nn.Module):
    """Landmark detection model based on Insightface's Retinaface model.

    Parameters
    ----------
    model_name : str
        Name of the landmark model from Insightface that should be used; options are
        '2d106det' (106 landmarks, 2D) or '1k3d68' (68 landmarks, 3D)
    detector : BaseDetector
        Which detector to use; options are ``SCRFDetector`` or ``YunetDetector``
    device : str
        Either 'cuda' (GPU) or 'cpu'
    """
    def __init__(self, model_name="2d106det", detector=SCRFDetector, device=DEVICE):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._det_model = detector(device=device)
        self._model = self._init_model()
        self.to(device).eval()

    def _init_model(self):
        """Initializes the landmark model by loading the ONNX model from disk."""
        from ..data import get_external_data_config

        f_in = get_external_data_config('insightface_path') / f'{self.model_name}.onnx'
        return OnnxModel(f_in, self.device)

    def forward(self, imgs):
        """Runs the landmark model on a set of images.

        Parameters
        ----------
        imgs : list, str, Path, torch.tensor
            Either a list of images, a path to a directory containing images, or an
            already loaded torch.tensor of shape N (batch) x C x H x W

        Returns
        -------
        out_lms : dict
            Dictionary with the following keys: 'lms' (landmarks), 'conf' (confidence),
            'img_idx' (image index), 'bbox' (bounding box)
        """

        out_det = self._det_model(imgs)

        if out_det.get("conf", None) is None:
            out_lms = {**out_det, "lms": None}
            return out_lms

        n_det = out_det["lms"].shape[0]
        bbox = out_det["bbox"]
        imgs_stack = imgs[out_det["img_idx"]]

        bw, bh = (bbox[:, 2] - bbox[:, 0]), (bbox[:, 3] - bbox[:, 1])
        center = torch.stack(
            [(bbox[:, 2] + bbox[:, 0]) / 2, (bbox[:, 3] + bbox[:, 1]) / 2], dim=1
        )

        onnx_input_shape = self._model._params["in_shapes"][0]
        scale = onnx_input_shape[3] / (torch.maximum(bw, bh) * 1.5)

        # Construct initial crop matrix based on rough bounding box,
        # then crop images to size 192 x 192
        M = torch.eye(3, device=self.device).repeat(n_det, 1, 1) * scale[:, None, None]
        M[:, 2, 2] = 1
        M[:, :2, 2] = -1 * center * scale[:, None] + onnx_input_shape[3] / 2
        imgs_crop = warp_affine(imgs_stack, M[:, :2, :], dsize=onnx_input_shape[2:])

        # Need to set batch dimension in output shape
        self._model._params["out_shapes"][0][0] = n_det
        lms = self._model.run(imgs_crop)["fc1"]  # fc1 = output (layer) name

        if lms.shape[1] == 3309:  # 3D data!
            # Reshape to n_det x n_lms x 3
            lms = lms.reshape((n_det, -1, 3))
            lms = lms[:, -68:, :]
        else:  # 2D data!
            # Reshape to n_det x n_lms x 2
            lms = lms.reshape((n_det, -1, 2))

        # Convert to cropped image coordinates
        lms[:, :, :2] = (lms[:, :, :2] + 1) * (onnx_input_shape[3] // 2)
        if lms.shape[2] == 3:
            lms[:, :, 2] *= onnx_input_shape[3] // 2

        lms = lms[:, :, :2]  # don't need 3rd dim

        # Warp landmarks from initial crop space (192 x 192) to
        # the original image space (?, ?)
        lms = transform_points(torch.inverse(M), lms)
        out_lms = {**out_det, "lms": lms}
        return out_lms

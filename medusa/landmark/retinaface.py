"""A landmark detection model based on Insightface's Retinaface model, but implemented
in PyTorch (but parts of it run with ONNX model), so can be fully run on GPU (no numpy
necessary)."""

import torch
from torch import nn

from ..onnx import OnnxModel
from ..crop import InsightfaceBboxCropModel
from ..defaults import DEVICE

from kornia.geometry.linalg import transform_points


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
    def __init__(self, model_name="2d106det", model_path=None, device=DEVICE):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._crop_model = InsightfaceBboxCropModel(output_size=(192, 192), device=device)
        self._model = self._init_model(model_path)
        self.to(device).eval()

    def _init_model(self, model_path):
        """Initializes the landmark model by loading the ONNX model from disk."""

        if model_path is None:
            from ..data import get_external_data_config
            model_path = get_external_data_config('insightface_path') / f'{self.model_name}.onnx'
        else:
            # Set model_name to name of the actual supplied model path
            self.model_name = model_path.stem.split('.')[0]

        return OnnxModel(model_path, self.device)

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

        out_crop = self._crop_model(imgs)
        if out_crop['imgs_crop'] is None:
            return {**out_crop, 'lms': None}

        imgs_crop = out_crop["imgs_crop"].float()
        n_det = imgs_crop.shape[0]
        input_shape = (192, 192)

        # Need to set batch dimension in output shape
        #self._model._params["out_shapes"][0][0] = n_det
        # august 2024: this cannot be run in batches for some reason
        #lms = self._model.run(imgs_crop[[0], :, :, :])["fc1"]  # fc1 = output (layer) name
        lms = []
        for i in range(n_det):
            lms.append(self._model.run(imgs_crop[[i], :, :, :])["fc1"])

        lms = torch.cat(lms, dim=0)

        if lms.shape[1] == 3309:  # 3D data!
            # Reshape to n_det x n_lms x 3
            lms = lms.reshape((n_det, -1, 3))
            lms = lms[:, -68:, :]
        else:  # 2D data!
            # Reshape to n_det x n_lms x 2
            lms = lms.reshape((n_det, -1, 2))

        # Convert to cropped image coordinates
        lms[:, :, :2] = (lms[:, :, :2] + 1) * (input_shape[1] // 2)
        if lms.shape[2] == 3:
            lms[:, :, 2] *= input_shape[1] // 2

        lms = lms[:, :, :2]  # don't need 3rd dim

        # Warp landmarks from initial crop space (192 x 192) to
        # the original image space (?, ?)
        crop_mat = out_crop["crop_mat"]
        lms = transform_points(torch.inverse(crop_mat), lms)
        out_lms = {**out_crop, "lms": lms}

        return out_lms

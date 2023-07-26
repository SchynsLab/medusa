"""A face recognition model based on Insightface's Retinaface model, but implemented
in PyTorch (but parts of it run with ONNX model), so can be fully run on GPU (no numpy
necessary)."""

from torch import nn

from ..onnx import OnnxModel
from ..crop import AlignCropModel
from ..defaults import DEVICE


class RetinafaceRecognitionModel(nn.Module):
    """Face recognition model based on Insightface's Retinaface model (trained using
    partial FC).

    Parameters
    ----------
    device : str
        Either 'cuda' (GPU) or 'cpu'
    """
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self._crop_model = AlignCropModel(device=device)
        self._model = self._init_model()
        self.to(device).eval()

    def _init_model(self):
        """Initializes the landmark model by loading the ONNX model from disk."""
        from ..data import get_external_data_config

        isf_path = get_external_data_config('insightface_path')
        if isf_path.stem == 'buffalo_l':
            f_in = isf_path / 'w600k_r50.onnx'
        else:
            f_in = isf_path / 'glintr100.onnx'

        return OnnxModel(f_in, self.device)

    def forward(self, imgs):
        """Runs the recognition model on a set of images.

        Parameters
        ----------
        imgs : list, str, Path, torch.tensor
            Either a list of images, a path to a directory containing images, or an
            already loaded torch.tensor of shape N (batch) x C x H x W

        Returns
        -------
        X_emb : torch.tensor
            Face embeddings of shape N x 512 (where N is the number of detections, not
            necessarily the number of input images)
        """
        imgs_crop = self._crop_model(imgs)['imgs_crop']
        if imgs_crop is None:
            raise ValueError("No faces detected in image(s)!")

        # By default, the ONNX model expects to return a single embedding of shape
        # (1, 512), so adjust the output shape to match the number of faces detected
        self._model._params['out_shapes'][0][0] = imgs_crop.shape[0]

        # Normalize images
        imgs_crop = (imgs_crop - 127.5) / 127.5

        # Run model on cropped+normalized images
        X_emb = self._model.run(imgs_crop)['1333']

        return X_emb

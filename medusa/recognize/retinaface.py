"""A face recognition model based on Insightface's Retinaface model, but implemented
in PyTorch (but parts of it run with ONNX model), so can be fully run on GPU (no numpy
necessary)."""
import torch
from torch import nn

from ..onnx import OnnxModel
from ..crop import AlignCropModel, InsightfaceBboxCropModel
from ..defaults import DEVICE


class RetinafaceRecognitionModel(nn.Module):
    """Face recognition model based on Insightface's Retinaface model (trained using
    partial FC).

    Parameters
    ----------
    model_path : str, Path
        Path to the ONNX model file; if None, will use the default model
    device : str
        Either 'cuda' (GPU) or 'cpu'
    """
    def __init__(self, model_path=None, device=DEVICE):
        super().__init__()
        self.device = device
        self._crop_model = AlignCropModel(device=device)
        self._model = self._init_model(model_path)
        self.to(device).eval()

    def _init_model(self, model_path):
        """Initializes the landmark model by loading the ONNX model from disk."""
        if model_path is None:
            from ..data import get_external_data_config
            isf_path = get_external_data_config('insightface_path')
            if isf_path.stem == 'buffalo_l':
                model_path = isf_path / 'w600k_r50.onnx'
            else:
                model_path = isf_path / 'glintr100.onnx'

        return OnnxModel(model_path, self.device)

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
        #self._model._params['out_shapes'][0][0] = imgs_crop.shape[0]

        # Normalize images
        imgs_crop = (imgs_crop - 127.5) / 127.5

        # Run model on cropped+normalized images
        X_emb = []
        for i in range(imgs_crop.shape[0]):
            inp_ = imgs_crop[[i], ...]
            X_emb.append(self._model.run(inp_, outputs_as_list=True)[0])

        X_emb = torch.cat(X_emb, dim=0)
        return X_emb


class RetinafaceGenderAgeModel(nn.Module):
    """Gender and age prediction model based on Insightface's model.

    Parameters
    ----------
    model_path : str, Path
        Path to the ONNX model file; if None, will use the default model
    device : str
        Either 'cuda' (GPU) or 'cpu'
    """
    def __init__(self, model_path=None, device=DEVICE):
        super().__init__()
        self.device = device
        self._crop_model = InsightfaceBboxCropModel(output_size=(96, 96), device=device)
        self._model = self._init_model(model_path)
        self.to(device).eval()

    def _init_model(self, model_path):
        """Initializes the model by loading the ONNX model from disk."""
        if model_path is None:
            from ..data import get_external_data_config
            isf_path = get_external_data_config('insightface_path')
            model_path = isf_path / 'genderage.onnx'

        return OnnxModel(model_path, self.device)

    def forward(self, imgs):
        """Runs the gender+age prediction model on a set of images.

        Parameters
        ----------
        imgs : torch.tensor
            A torch.tensor of shape N (batch) x C x H x W

        Returns
        -------
        attr : dict
            Dictionary with the following keys: 'gender' (0 = female, 1 = male),
            'age' (age in years), both torch.tensors with the first dimension being the
            number of faces detected in the input images
        """

        out_crop = self._crop_model(imgs)

        if out_crop.get("imgs_crop", None) is None:
            return {**out_crop, "gender": None, "age": None}

        imgs_crop = out_crop["imgs_crop"].float()
        n_det = imgs_crop.shape[0]

        # Need to set batch dimension in output shape
        #self._model._params["out_shapes"][0][0] = n_det
        out = []
        for i in range(n_det):
            out.append(self._model.run(imgs_crop[[i], ...], outputs_as_list=True)[0])

        out = torch.cat(out, dim=0)
        gender = torch.argmax(out[:, :2], dim=1)  # 0 = female, 1 = male
        age = (out[:, 2] * 100).round().int()

        attr = {'gender': gender, 'age': age}

        return attr

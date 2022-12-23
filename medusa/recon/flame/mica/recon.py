from collections import OrderedDict

import torch
import torch.nn.functional as F

from ....defaults import DEVICE
from ....io import load_inputs
from ..base import FlameReconModel
from ..decoders import FLAME
from .encoders import Arcface, MappingNetwork


class MicaReconModel(FlameReconModel):
    """A simplified implementation of the MICA 3D face reconstruction model
    (https://zielon.github.io/mica/), for inference only.

    Parameters
    ----------
    device : str
        Either 'cuda' (uses GPU) or 'cpu'
    """

    def __init__(self, device=DEVICE):
        """Initializes a MicaReconModel object."""
        self.device = device
        self._load_cfg()  # method inherited from parent
        self._create_submodels()
        self._load_submodels()

    def __str__(self):
        return "mica"

    def _create_submodels(self):
        """Loads the submodels associated with MICA. To summarizes:

        - `E_arcface`: predicts a 512-D embedding for a (cropped, 112x112) image
        - `E_flame`: predicts (coarse) FLAME parameters given a 512-D embedding
        - `D_flame`: outputs a ("coarse") mesh given shape FLAME parameters
        """
        self.E_arcface = Arcface().to(self.device)
        self.E_arcface.eval()
        self.E_flame = MappingNetwork(512, 300, 300).to(self.device)
        self.E_flame.eval()
        self.D_flame = FLAME(self.cfg["flame_path"], n_shape=300, n_exp=0).to(
            self.device
        )
        self.D_flame.eval()

    def _load_submodels(self):
        """Loads the weights for the Arcface submodel as well as the
        MappingNetwork that predicts FLAME shape parameters from the Arcface
        output."""
        checkpoint = torch.load(self.cfg["mica_path"], map_location=self.device)
        self.E_arcface.load_state_dict(checkpoint["arcface"])

        # The original weights also included the data for the FLAME model (template
        # vertices, faces, etc), which we don't need here, because we use a common
        # FLAME decoder model (in decoders.py)
        new_checkpoint = OrderedDict()
        for key, value in checkpoint["flameModel"].items():
            # The actual mapping-network weights are stored in keys starting with
            # regressor.
            if "regressor." in key:
                new_checkpoint[key.replace("regressor.", "")] = value

        self.E_flame.load_state_dict(new_checkpoint)

    def _encode(self, image):
        """Encodes an image into a set of FLAME shape parameters."""
        out_af = self.E_arcface(image)  # output of arcface
        out_af = F.normalize(out_af)
        shape_code = self.E_flame(out_af)
        return shape_code

    def _decode(self, shape_code):
        """Decodes the shape code into a set of vertices following the (coarse)
        FLAME topology."""
        v, _ = self.D_flame(shape_code)
        return v

    def get_cam_mat(self):

        cam_mat = torch.eye(4) * 8
        cam_mat[3, 3] = 1
        cam_mat[2, 3] = 4
        return cam_mat

    def __call__(self, imgs):
        """Performs 3D reconstruction on the supplied image.

        Parameters
        ----------
        image : np.ndarray, torch.Tensor
            Ideally, a numpy array or torch tensor of shape 1 x 3 x 112 x 112
            (1, C, W, H), representing a cropped image as done by the
            InsightFaceCroppingModel

        Returns
        -------
        out : dict
            A dictionary with two keys: ``"v"``, the reconstructed vertices (5023 in
            total) and ``"mat"``, a 4x4 Numpy array representing the local-to-world
            matrix, which is in the case of MICA the identity matrix
        """
        imgs = load_inputs(
            imgs,
            load_as="torch",
            channels_first=True,
            with_batch_dim=True,
            device=self.device,
        )
        imgs = self._preprocess(imgs)
        shape_code = self._encode(imgs)
        v = self._decode(shape_code)
        out = {"v": v, "mat": None}

        return out

"""Decoder-modules for FLAME-based reconstruction models.

See ./deca/license.md for conditions for use.
"""

import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lbs import lbs


class FLAME(nn.Module):
    """Generates a FLAME-based based on latent parameters."""

    def __init__(self, model_path, n_shape, n_exp):
        super().__init__()
        # print("creating the FLAME Decoder")
        with open(model_path, "rb") as f:
            # Silence scipy DeprecationWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                model = pickle.load(f, encoding="latin1")

        self.faces = _to_np(model['f'], dtype=np.int64)
        self.register_buffer("faces_tensor", _to_tensor(self.faces, dtype=torch.long))

        # The vertices of the template model
        self.register_buffer(
            "v_template", _to_tensor(_to_np(model['v_template']), dtype=torch.float32)
        )
        # The shape components and expression
        shapedirs = _to_tensor(_to_np(model['shapedirs']), dtype=torch.float32)
        shapedirs = torch.cat(
            [shapedirs[:, :, :n_shape], shapedirs[:, :, 300 : (300 + n_exp)]], 2
        )
        self.register_buffer("shapedirs", shapedirs)

        # The pose components
        num_pose_basis = model['posedirs'].shape[-1]
        posedirs = np.reshape(model['posedirs'], [-1, num_pose_basis]).T
        self.register_buffer("posedirs", _to_tensor(_to_np(posedirs), dtype=torch.float32))

        self.register_buffer(
            "J_regressor", _to_tensor(_to_np(model['J_regressor']), dtype=torch.float32)
        )
        parents = _to_tensor(_to_np(model['kintree_table'][0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights", _to_tensor(_to_np(model['weights']), dtype=torch.float32)
        )

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=torch.float32, requires_grad=False)
        self.register_parameter(
            "eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False)
        )
        default_neck_pose = torch.zeros([1, 3], dtype=torch.float32, requires_grad=False)
        self.register_parameter(
            "neck_pose", nn.Parameter(default_neck_pose, requires_grad=False)
        )

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        pose_params=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        eye_pose_params = self.eye_pose.expand(batch_size, -1)

        if expression_params is None:
            betas = shape_params
        else:
            betas = torch.cat([shape_params, expression_params], dim=1)

        if pose_params is None:
            pose_params = torch.zeros((batch_size, 6)).to(shape_params.device)

        full_pose = torch.cat(
            [
                pose_params[:, :3],
                self.neck_pose.expand(batch_size, -1),
                pose_params[:, 3:],
                eye_pose_params,
            ],
            dim=1,
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, T, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )

        return vertices, T


class FLAMETex(nn.Module):
    """FLAME texture:

    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    FLAME texture converted from BFM:
    https://github.com/TimoBolkart/BFM_to_FLAME
    """

    def __init__(self, model_path=None, n_tex=50):
        super().__init__()
        self.n_tex = n_tex
        self.tex_space = self._load_model(model_path)
        self._register_buffers(self.tex_space)

    def _load_model(self, tex_model_path):
        # Avoids circular import
        from ...data import get_external_data_config

        if tex_model_path is None:
            ext_data_path = get_external_data_config('flame_path').parent
            tex_model_path = ext_data_path / "FLAME_albedo_from_BFM.npz"

        if not tex_model_path.is_file():
            raise ValueError("Couldn't find tex model at {tex_model_path}!")

        return np.load(tex_model_path)

    def _register_buffers(self, tex_space):
        texture_mean = tex_space["MU"].reshape(1, -1)
        texture_basis = tex_space["PC"][:, :self.n_tex]  # 199 comp

        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis).float()[None, ...]
        self.register_buffer("texture_mean", texture_mean)
        self.register_buffer("texture_basis", texture_basis)

    def forward(self, texcode):
        """
        texcode: [batchsize, n_tex]
        texture: [bz, 3, 256, 256], range: 0-1
        """
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture


def _to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def _to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

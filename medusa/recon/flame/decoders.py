# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lbs import lbs


class FLAME(nn.Module):
    """borrowed from
    https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py Given
    flame parameters this class generates a differentiable FLAME function which
    outputs the a mesh and 2D/3D facial landmarks."""

    def __init__(self, model_path, n_shape, n_exp):
        super().__init__()
        # print("creating the FLAME Decoder")
        with open(model_path, "rb") as f:
            # Silence scipy DeprecationWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                ss = pickle.load(f, encoding="latin1")

            flame_model = Struct(**ss)

        self.faces = to_np(flame_model.f, dtype=np.int64)
        self.dtype = torch.float32
        self.register_buffer("faces_tensor", to_tensor(self.faces, dtype=torch.long))

        # The vertices of the template model
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [shapedirs[:, :, :n_shape], shapedirs[:, :, 300 : (300 + n_exp)]], 2
        )
        self.register_buffer("shapedirs", shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            "eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False)
        )
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
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

    def __init__(self, model_path, n_tex):
        super(FLAMETex, self).__init__()
        tex_space = np.load(model_path)
        texture_mean = tex_space["MU"].reshape(1, -1)
        texture_basis = tex_space["PC"].reshape(-1, 199)  # 199 comp

        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
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


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

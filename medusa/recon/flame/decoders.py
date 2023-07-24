"""Decoder-modules for FLAME-based reconstruction models.

See ./deca/license.md for conditions for use.
"""

import pickle
import warnings
from pathlib import Path

from scipy.sparse import csc_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from chumpy.ch import Ch
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, euler_angles_to_matrix

from .lbs import lbs
from ...defaults import DEVICE


class FlameShape(nn.Module):
    """Generates a FLAME-based mesh (shape only) from 3DMM parameters.

    Parameters
    ----------
    """

    def __init__(self, n_shape=300, n_expr=100, parameters=None, device=DEVICE,
                 **init_parameters):
        """Initializes the FLAME decoder."""
        super().__init__()

        self.n_shape = n_shape
        self.n_expr = n_expr
        self.parameters_ = [] if parameters is None else parameters

        model = _load_flame_model()
        self.register_buffer('tris', _to_tensor(model['f']))
        self.register_buffer('v_template', _to_tensor(model['v_template']))

        # The shape components and expression
        shape_dirs = _to_tensor(model['shapedirs'])
        shape_dirs = torch.cat([shape_dirs[:, :, :n_shape], shape_dirs[:, :, 300:(300 + n_expr)]], dim=2)
        self.register_buffer('shape_dirs', shape_dirs)

        # The pose components
        num_poses = model['posedirs'].shape[2]
        pose_dirs = _to_tensor(model['posedirs'].reshape([-1, num_poses]).T)
        self.register_buffer('pose_dirs', pose_dirs)

        self.register_buffer('J_regressor', _to_tensor(model['J_regressor']))
        parents = _to_tensor(model['kintree_table'][0])
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights', _to_tensor(model['weights']))

        batch_size = 1

        self.shape = nn.Parameter(torch.zeros([batch_size, n_shape]))
        self.expr = nn.Parameter(torch.zeros((batch_size, n_expr)))
        self.eye_pose = nn.Parameter(torch.zeros([batch_size, 6]))
        self.neck_pose = nn.Parameter(torch.zeros([batch_size, 3]))
        self.jaw_pose = nn.Parameter(torch.zeros([batch_size, 3]))
        self.global_pose = nn.Parameter(torch.zeros([batch_size, 3]))

        for p_name, p in init_parameters.items():
            setattr(self, p_name, nn.Parameter(p))

        for p_name, p in self.named_parameters():
            # nn.Parameter objects are always trainable by default, but we start with
            # the assumption that they're not trainable
            if p_name not in self.parameters_:
                p.requires_grad_(False)

        self.to(device)

    def _has_parameter(self, param_name):

        for name, _ in self.named_parameters():
            if name == param_name:
                return True

        return False

    def get_full_pose(self):
        """Returns the full pose vector."""
        return torch.cat([self.global_pose, self.neck_pose, self.jaw_pose, self.eye_pose], dim=1)

    def forward(self, batch_size=None, **inputs):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """

        if not inputs and batch_size is None:
            # We need to infer batch size from either the inputs or from the explicitly
            # set batch_size (in case there are no inputs at all)
            raise ValueError("Either inputs or batch_size must be provided!")
        elif batch_size is None:
            # Infer batch_size from inputs (any will do; assumed to be Bx{dim})
            batch_size = next(iter(inputs.values())).shape[0]

        # Fix existing parameters
        poses = ['global_pose', 'neck_pose', 'jaw_pose', 'eye_pose']
        for param in ['shape', 'expr'] + poses:
            if param in inputs and param in self.parameters_:
                # param should be either part of the inputs or of the trainable parameters
                raise ValueError(f"Parameter {param} is also part of inputs!")

            if param not in inputs:
                existing_param = self.get_parameter(param)
                if existing_param.shape[0] != batch_size:
                    # Fix batch size
                    existing_param = existing_param.expand(batch_size, -1)

                inputs[param] = existing_param

        shape_expr = torch.cat([inputs['shape'], inputs['expr']], dim=1)
        full_pose = torch.cat([*[inputs.get(k) for k in poses]], dim=1)
        v_template = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        v, R, _ = lbs(shape_expr, full_pose, v_template, self.shape_dirs, self.pose_dirs,
                      self.J_regressor, self.parents, self.lbs_weights)

        return v, R


class FlameLandmark(nn.Module):

    def __init__(self, lm_type='68', lm_dim='2d', device=DEVICE):
        super().__init__()
        self.lm_type = lm_type
        self.lm_dim = lm_dim
        self._load_data_and_register_buffers()
        self.device = device
        self.to(device)

    def _load_data_and_register_buffers(self):

        flame_model = _load_flame_model()
        self.register_buffer('tris', _to_tensor(flame_model['f']))

        parents = flame_model['kintree_table'][0]
        parents[0] = -1

        # kinetic chain
        neck_kin_chain = []
        curr_idx = 1
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = parents[curr_idx]

        neck_kin_chain = torch.as_tensor(neck_kin_chain)
        self.register_buffer('neck_kin_chain', neck_kin_chain)

        if self.lm_type == '68':
            lms_path = Path(__file__).parents[2] / 'data' / 'flame' / 'landmark_embedding.npy'
            lms = np.load(lms_path, allow_pickle=True)[()]
            lms['lmk_faces_idx'] = lms.pop('static_lmk_faces_idx')
            lms['lmk_bary_coords'] = lms.pop('static_lmk_bary_coords')
        elif self.lm_type == 'mp':
            lms_path = Path(__file__).parents[2] / 'data' / 'flame' / 'mediapipe_landmark_embedding.npz'
            lms = dict(np.load(lms_path))
            lms['lmk_faces_idx'] = lms.pop('lmk_face_idx').astype(np.int64)
            lms['lmk_bary_coords'] = lms.pop('lmk_b_coords')
        else:
            raise ValueError("Choose `lm_type` from '68' or 'mp'")

        lms['lmk_faces_idx'] = torch.tensor(lms['lmk_faces_idx'], dtype=torch.long)
        lms['lmk_bary_coords'] = torch.tensor(lms['lmk_bary_coords'], dtype=torch.float32)
        self.register_buffer('lmk_faces_idx', lms['lmk_faces_idx'])
        self.register_buffer('lmk_bary_coords', lms['lmk_bary_coords'])

        if self.lm_type == '68':
            #lms['dynamic_lmk_faces_idx'] = torch.tensor(lms['dynamic_lmk_faces_idx'], dtype=torch.long)
            #lms['dynamic_lmk_bary_coords'] = torch.tensor(lms['dynamic_lmk_bary_coords'], dtype=torch.float32)
            self.register_buffer('dynamic_lmk_faces_idx', lms['dynamic_lmk_faces_idx'])
            self.register_buffer('dynamic_lmk_bary_coords', lms['dynamic_lmk_bary_coords'])
        else:
            self.register_buffer('landmark_indices', _to_tensor(lms['landmark_indices']))

    def forward(self, v, poses):

        batch_size = v.shape[0]
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        if self.lm_type == 'mp':
            lmk_faces_idx = lmk_faces_idx.contiguous()
            lmk_bary_coords = lmk_bary_coords.contiguous()

        if self.lm_type == '68':
            chunks = []
            for chunk in torch.chunk(poses, poses.shape[1] // 3, dim=1):
                chunks.append(matrix_to_rotation_6d(euler_angles_to_matrix(chunk, 'XYZ')))

            poses = torch.cat(chunks, dim=1)
            with torch.no_grad():
                dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
                    poses
                )

            dyn_lmk_faces_idx = dyn_lmk_faces_idx.expand(batch_size, -1)
            dyn_lmk_bary_coords = dyn_lmk_bary_coords.expand(batch_size, -1, -1)

            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        lmk_tris = torch.index_select(self.tris, 0, lmk_faces_idx.view(-1).to(torch.long)).view(batch_size, -1, 3)
        lmk_tris += torch.arange(batch_size, dtype=torch.long, device=self.device).view(-1, 1, 1) * v.shape[1]
        lmk_v = v.view(-1, 3)[lmk_tris].view(batch_size, -1, 3, 3)
        lmk = torch.einsum('blfi,blf->bli', [lmk_v, lmk_bary_coords])

        return lmk

    def _find_dynamic_lmk_idx_and_bcoords(self, pose):

        batch_size = pose.shape[0]
        aa_pose = torch.index_select(pose.view(batch_size, -1, 6), 1, self.neck_kin_chain)
        rot_mats = rotation_6d_to_matrix(aa_pose.view(-1, 6)).view([batch_size, -1, 3, 3])

        rel_rot_mat = torch.eye(3, device=self.device).unsqueeze(dim=0).expand(batch_size, -1, -1)

        for idx in range(len(self.neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(torch.clamp(-self._rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(self.dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_bary_coords = torch.index_select(self.dynamic_lmk_bary_coords, 0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_bary_coords

    def _rot_mat_to_euler(self, rot_mats):
        # Calculates rotation matrix to euler angles
        # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

        sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                        rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
        return torch.atan2(-rot_mats[:, 2, 0], sy)


class FlameTex(nn.Module):
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


def _load_flame_model():

    from ...data import get_external_data_config
    model_path = get_external_data_config('flame_path')

    with open(model_path, "rb") as f:
        # Silence scipy DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = pickle.load(f, encoding="latin1")

    return model


def _to_tensor(x):
    """Converts a numpy array to a torch tensor."""
    if isinstance(x, Ch):
        x = np.asarray(x)

    if isinstance(x, csc_matrix):
        x = x.todense()

    if x.dtype in (np.uint32, np.int32):
        x = x.astype(np.int64)

    elif x.dtype == np.float64:
        x = x.astype(np.float32)

    x = torch.from_numpy(x)
    x.requires_grad_(False)

    return x

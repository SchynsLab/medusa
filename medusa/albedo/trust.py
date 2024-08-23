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

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image

from ..defaults import DEVICE
from ..recon.flame.deca.encoders import Resnet50Encoder
from ..recon.flame.decoders import FlameTex


class TRUST(nn.Module):
    def __init__(self, device=DEVICE):
        # avoid circular import
        from ..data import get_external_data_config
        super().__init__()
        self.device = device
        self._cfg = get_external_data_config()
        self._create_submodels()
        self.to(device).eval()

    def _create_submodels(self):

        self.param_dict = {
            'n_tex': 54,
            'n_light': 27,
            'n_scenelight': 3,
            'n_facelight': 27
        }

        self.E_albedo = Resnet50Encoder(outsize=self.param_dict['n_tex'], version='v2').to(self.device)
        checkpoint = torch.load(self._cfg['trust_albedo_encoder_path'], map_location=self.device)
        self.E_albedo.load_state_dict(checkpoint["E_albedo"])

        self.E_scene_light = Resnet50Encoder(outsize=self.param_dict['n_scenelight']).to(self.device)
        checkpoint = torch.load(self._cfg['trust_scene_light_encoder_path'], map_location=self.device)
        self.E_scene_light.load_state_dict(checkpoint["E_scene_light"])

        self.E_face_light = Resnet50Encoder(outsize=self.param_dict['n_facelight']).to(self.device)
        checkpoint = torch.load(self._cfg['trust_face_light_encoder_path'], map_location=self.device)
        self.E_face_light.load_state_dict(checkpoint["E_face_light"])

        # decoding
        self.D_flame_tex = FlameTex(model_path=self._cfg['trust_albedo_decoder_path'],
                                    n_tex=54).to(self.device) # texture layer

    def _fuse_light(self, E_scene_light_pred, E_face_light_pred):
        
        normalized_sh_params = F.normalize(E_face_light_pred, p=1, dim=1)
        lightcode = E_scene_light_pred.unsqueeze(1).expand(-1, 9, -1) * normalized_sh_params

        return lightcode, E_scene_light_pred, normalized_sh_params

    def _encode(self, imgs, scene_imgs):
        '''
        :param images:
        :param scene_images:
        :param face_lighting:
        :param scene_lighting:
        :return:
        '''

        B, C, H, W = imgs.size()
        E_scene_light_pred = self.E_scene_light(scene_imgs)  # B x 3
        #E_face_light_pred = self.E_face_light(imgs).reshape(B, 9, 3)
        E_scene_light_pred = E_scene_light_pred[..., None, None].repeat(1, 1, H, W)
        
        imgs_cond = torch.cat((E_scene_light_pred, imgs), dim=1)
        tex_code = self.E_albedo(imgs_cond)
        
        return tex_code

    def _decode(self, tex_code):
        albedo = self.D_flame_tex(tex_code)
        return albedo

    def forward(self, imgs, scene_imgs):
        
        if imgs.dtype == torch.uint8:
            imgs = imgs.float()

        if imgs.max() >= 1.0:
            imgs = imgs.div(255.0)

        if scene_imgs.dtype == torch.uint8:
            scene_imgs = scene_imgs.float()

        if scene_imgs.max() >= 1.0:
            scene_imgs = scene_imgs.div(255.0)

        tex_code = self._encode(imgs, scene_imgs)
        albedo = self._decode(tex_code)

        
        out = {'albedo': albedo}

        return out

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
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .rasterizer import StandardRasterizer
from ..recon.emoca import utils


class SRenderY(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256, device='cuda'):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        self.device = device
        self.rasterizer = StandardRasterizer(image_size, device=device)        

        verts, uvcoords, faces, uvfaces = utils.load_obj(obj_filename)
        verts = verts[None, ...]
        uvcoords = uvcoords[None, ...]
        faces = faces[None, ...]
        uvfaces = uvfaces[None, ...]

        # faces
        dense_triangles = utils.generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None,:,:])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = utils.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max()+1, 1).float()/255.
        colors = colors.to(device=device)
        face_colors = utils.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def render_shape(self, vertices, transformed_vertices, colors = None, images=None, detail_normal_images=None, 
                lights=None, return_grid=False, uv_detail_normals=None, h=None, w=None):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        # set lighting
        if lights is None:
            light_positions = torch.tensor(
                [
                [-1,1,1],
                [1,1,1],
                [-1,-1,1],
                [1,-1,1],
                [0,0,1]
                ]
            )[None,:,:].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float()*1.7
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10

        # Attributes
        face_vertices = utils.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = utils.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)); face_normals = utils.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = utils.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1)); transformed_face_normals = utils.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        
        colors = self.face_colors.expand(batch_size, -1, -1, -1)
        attributes = torch.cat([colors, 
                        transformed_face_normals.detach(), 
                        face_vertices.detach(), 
                        face_normals,
                        self.face_uvcoords.expand(batch_size, -1, -1, -1)], 
                        -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0.15).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images

        alpha_images = alpha_images*pos_mask
        if images is None:
            shape_images = shaded_images*alpha_images + torch.zeros_like(shaded_images).to(vertices.device)*(1-alpha_images)
        else:
            shape_images = shaded_images*alpha_images + images*(1-alpha_images)

        return shape_images
    
    
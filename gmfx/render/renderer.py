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
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread

from ..recon import utils
from standard_rasterize_cuda import standard_rasterize
        

# Note to self: `torch` always needs to be imported before
# importing `standard_rasterize_cuda`, because importing `torch`
# will make libc10.so available.


class StandardRasterizer(nn.Module):
    """ Alg: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation
    Notice:
        x,y,z are in image space, normalized to [-1, 1]
        can render non-squared image
        not differentiable
    """
    def __init__(self, height, width=None):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        if width is None:
            width = height
        self.h = h = height; self.w = w = width

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        device = vertices.device
        if h is None:
            h = self.h
        if w is None:
            w = self.h; 
        bz = vertices.shape[0]
        depth_buffer = torch.zeros([bz, h, w]).float().to(device) + 1e6
        triangle_buffer = torch.zeros([bz, h, w]).int().to(device) - 1
        baryw_buffer = torch.zeros([bz, h, w, 3]).float().to(device)
        vert_vis = torch.zeros([bz, vertices.shape[1]]).float().to(device)
        vertices = vertices.clone().float()
        
        vertices[...,:2] = -vertices[...,:2]
        vertices[...,0] = vertices[..., 0]*w/2 + w/2
        vertices[...,1] = vertices[..., 1]*h/2 + h/2
        vertices[...,0] = w - 1 - vertices[..., 0]
        vertices[...,1] = h - 1 - vertices[..., 1]
        vertices[...,0] = -1 + (2*vertices[...,0] + 1)/w
        vertices[...,1] = -1 + (2*vertices[...,1] + 1)/h
        #
        vertices = vertices.clone().float()
        vertices[...,0] = vertices[..., 0]*w/2 + w/2 
        vertices[...,1] = vertices[..., 1]*h/2 + h/2 
        vertices[...,2] = vertices[..., 2]*w/2
        f_vs = utils.face_vertices(vertices, faces)

        standard_rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)
        pix_to_face = triangle_buffer[:,:,:,None].long()
        bary_coords = baryw_buffer[:,:,:,None,:]
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        return pixel_vals


class SRenderY(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        self.rasterizer = StandardRasterizer(image_size)
        self.uv_rasterizer = StandardRasterizer(uv_size)
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
        face_colors = utils.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)
    
    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point', h=None, w=None):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights: 
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        # attributes
        face_vertices = utils.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = utils.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)); face_normals = utils.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = utils.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1)); transformed_face_normals = utils.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), 
                                transformed_face_normals.detach(), 
                                face_vertices.detach(), 
                                face_normals], 
                                -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h=h, w=w)
        
        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]; grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type=='point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
                else:
                    shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
            images = albedo_images*shading_images
        else:
            images = albedo_images
            shading_images = images.detach()*0.

        outputs = {
            'images': images*alpha_images,
            'albedo_images': albedo_images*alpha_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images*alpha_images,
            'transformed_normals': transformed_normals,
        }
        
        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
                N[:,0]*0.+1., N[:,0], N[:,1], \
                N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
                N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
                ], 
                1) # [bz, 9, h, w]
        sh = sh*self.constant_factor[None,:,None,None]
        shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

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
        if colors is None:
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
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images

        alpha_images = alpha_images*pos_mask
        if images is None:
            shape_images = shaded_images*alpha_images + torch.zeros_like(shaded_images).to(vertices.device)*(1-alpha_images)
        else:
            shape_images = shaded_images*alpha_images + images*(1-alpha_images)
        if return_grid:
            uvcoords_images = rendering[:, 12:15, :, :]; 
            grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
            return shape_images, normal_images, grid, alpha_images
        else:
            return shape_images
    
    def render_depth(self, transformed_vertices):
        '''
        -- rendering depth
        '''
        batch_size = transformed_vertices.shape[0]

        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
        z = -transformed_vertices[:,:,2:].repeat(1,1,3).clone()
        z = z-z.min()
        z = z/z.max()
        # Attributes
        attributes = utils.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images
    
    def render_colors(self, transformed_vertices, colors):
        '''
        -- rendering colors: could be rgb color/ normals, etc
            colors: [bz, num of vertices, 3]
        '''
        batch_size = colors.shape[0]

        # Attributes
        attributes = utils.face_vertices(colors, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        ####
        alpha_images = rendering[:, [-1], :, :].detach()
        images = rendering[:, :3, :, :]* alpha_images
        return images

    def world2uv(self, vertices):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = utils.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1), self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        return uv_vertices

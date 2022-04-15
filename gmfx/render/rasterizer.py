import torch
import torch.nn as nn
# Note to self: `torch` always needs to be imported before
# importing `standard_rasterize_cuda`, because importing `torch`
# will make libc10.so available.
from standard_rasterize_cuda import standard_rasterize

from ..recon.emoca import utils


class StandardRasterizer(nn.Module):
    """ Alg: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation
    Notice:
        x,y,z are in image space, normalized to [-1, 1]
        can render non-squared image
        not differentiable
    """
    def __init__(self, height, width=None, device='cuda'):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        if width is None:
            width = height
        
        self.device = device   
        self.h = height
        self.w = width

    def forward(self, v, f, attrs=None, h=None, w=None):
        device = self.device

        if h is None:
            h = self.h

        if w is None:
            w = self.h; 

        bz = v.shape[0]
        depth_buffer = torch.zeros([bz, h, w]).float().to(device) + 1e6
        triangle_buffer = torch.zeros([bz, h, w]).int().to(device) - 1
        baryw_buffer = torch.zeros([bz, h, w, 3]).float().to(device)
        v = v.clone().float()
        
        v[..., :2] = -v[..., :2]
        v[..., 0] = v[..., 0] * w/2 + w/2
        v[..., 1] = v[..., 1] * h/2 + h/2
        v[..., 0] = w - 1 - v[..., 0]
        v[..., 1] = h - 1 - v[..., 1]
        v[..., 0] = -1 + (2 * v[...,0] + 1) / w
        v[..., 1] = -1 + (2 * v[...,1] + 1) / h

        v = v.clone().float()
        v[..., 0] = v[..., 0] * w/2 + w/2 
        v[..., 1] = v[..., 1] * h/2 + h/2 
        v[..., 2] = v[..., 2] * w/2
        f_vs = utils.face_vertices(v, f)

        standard_rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)
        pix_to_face = triangle_buffer[... , None].long()
        bary_coords = baryw_buffer[:, :, :, None, :]
        vismask = (pix_to_face > -1).float()
        D = attrs.shape[-1]
        attrs = attrs.clone()
        attrs = attrs.view(attrs.shape[0] * attrs.shape[1], 3, attrs.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attrs.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals

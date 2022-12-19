import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer import HardFlatShader, SoftPhongShader, HardPhongShader
from pytorch3d.renderer import FoVPerspectiveCameras, FoVOrthographicCameras
from pytorch3d.renderer import TexturesVertex, PointLights

from .. import DEVICE
from .base import BaseRenderer


class PytorchRenderer(BaseRenderer):

    def __init__(self, viewport, cam_mat=None, cam_type='orthographic', shading='flat', device=DEVICE):
        self.viewport = viewport
        self.device = device
        self._settings = self._setup_settings(viewport)
        self._lights = PointLights(device=device, location=[[0., 0., 3.0]])
        self._cameras = self._setup_cameras(cam_mat, cam_type)
        self._rasterizer = MeshRasterizer(self._cameras, self._settings)
        self._shader = self._setup_shader(shading)
        self._renderer = MeshRenderer(self._rasterizer, self._shader)

    def __str__(self):
        return 'pytorch3d'

    def _setup_settings(self, viewport):

        if isinstance(viewport, (list, tuple, np.ndarray)):
            # pytorch3d wants (H, W), but we always use (W, H) to
            # keep the renderer API consistent
            viewport_ = (viewport[1], viewport[0])
        else:
            viewport_ = (viewport, viewport)

        return RasterizationSettings(
            image_size=(int(viewport_[0]), int(viewport_[1])),
            blur_radius=0.0,
            faces_per_pixel=1
        )

    def _setup_cameras(self, cam_mat, cam_type):

        if cam_mat is None:
            # Mapping to pytorch3d space
            cam_mat = torch.tensor(device=self.device, data=[
                [-1., 0., 0.],
                [0., 1., 0.],
                [0., 0., -1.],
            ]).unsqueeze(0)  # 1 x 3 x 3

        if cam_type == 'orthographic':
            cam = FoVOrthographicCameras(device=self.device, T=[[0, 0, 4]])
        elif cam_type == 'perspective':
            fov = 2 * np.arctan(self.viewport[1] / (2 * self.viewport[0]))
            cam = FoVPerspectiveCameras(fov=fov, znear=1., zfar=10000., degrees=False, device=self.device,
                                        R=cam_mat)
        else:
            raise ValueError(f"Unknown camera type '{cam_type}'!")

        return cam

    def _setup_shader(self, shading):

        if shading == 'flat':
            shader = HardFlatShader(self.device, self._cameras, self._lights)
        elif shading == 'smooth':
            shader = HardPhongShader(self.device, self._cameras, self._lights)
        else:
            raise ValueError("For now, choose either 'flat' or 'smooth'!")

        return shader

    def __call__(self, v, tris, tex=None):

        v, tris = self._preprocess(v, tris, format='torch')

        if tex is None:
            tex = TexturesVertex(torch.ones_like(v, device=self.device))

        tris = tris.repeat(v.shape[0], 1, 1)
        meshes = Meshes(v, tris, tex)
        imgs = self._renderer(meshes)

        if imgs.shape[0] != 1:
            # TOFIX
            imgs = imgs.amax(dim=0, keepdim=True)

        imgs = (imgs * 255).to(torch.uint8)

        return imgs

    def close(self):
        pass

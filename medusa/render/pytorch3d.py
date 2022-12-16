import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer import HardFlatShader, SoftPhongShader
from pytorch3d.renderer import FoVPerspectiveCameras, FoVOrthographicCameras
from pytorch3d.renderer import TexturesVertex, PointLights

from .. import DEVICE


class PytorchRenderer:

    def __init__(self, viewport, cam_mat=None, cam_type='orthographic', shading='flat', device=DEVICE):
        self.device = device
        self._settings = RasterizationSettings(image_size=viewport)
        self._lights = PointLights(device=device, location=(0, 0, 1))
        self._cameras = self._setup_cameras(cam_mat, cam_type)
        self._rasterizer = MeshRasterizer(self._cameras, self._settings)
        self._shader = self._setup_shader(shading)
        self._renderer = MeshRenderer(self._rasterizer, self._shader)

    def _setup_cameras(self, cam_mat, cam_type):

        if cam_type == 'ortographic':
            cam = FoVOrthographicCameras(device=self.device)
        else:
            cam = FoVPerspectiveCameras(device=self.device)

        return cam

    def _setup_shader(self, shading):

        if shading == 'flat':
            shader = HardFlatShader(self.device, self._cameras, self._lights)
        else:
            shader = SoftPhongShader(self.device, self._cameras, self._lights)

        return shader

    def __call__(self, v, f, tex=None):

        if tex is None:
            tex = TexturesVertex(torch.ones_like(v, device=self.device))

        f = f.repeat(v.shape[0], 1, 1)
        meshes = Meshes(v, f, tex)
        imgs = self._renderer(meshes)

        return imgs

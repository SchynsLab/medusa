"""Module with a renderer class based on ``pytorch3d``."""

import numpy as np
import torch
from pytorch3d.renderer import (FoVOrthographicCameras, FoVPerspectiveCameras,
                                HardFlatShader, HardPhongShader,
                                MeshRasterizer, MeshRenderer, PointLights,
                                RasterizationSettings, TexturesVertex, BlendParams)
from pytorch3d.structures import Meshes

from ..defaults import DEVICE
from .base import BaseRenderer


class PytorchRenderer(BaseRenderer):
    """A pytorch3d-based renderer.
    
    Parameters
    ----------
    viewport : tuple[int]
        Desired output image size (width, height), in pixels; should match
        the original image (before cropping) that was reconstructed
    cam_mat : torch.tensor
        A camera matrix to set the position/angle of the camera
    cam_type : str
        Either 'orthographic' (for Flame-based reconstructions) or
        'perpective' (for mediapipe reconstructions)
    shading : str
        Type of shading ('flat', 'smooth', or 'wireframe')
    wireframe_opts : None, dict
        Dictionary with extra options for wireframe rendering (options: 'width', 'color')
    device : str
        Device to store the image on ('cuda' or 'cpu')
    """
    def __init__(
        self,
        viewport,
        cam_mat=None,
        cam_type="orthographic",
        shading="flat",
        device=DEVICE,
    ):
        """Initializes a PytorchRenderer object."""
        self.viewport = viewport
        self.device = device
        self._settings = self._setup_settings(viewport)
        self._lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
        self._cameras = self._setup_cameras(cam_mat, cam_type)
        self._rasterizer = MeshRasterizer(self._cameras, self._settings)
        self._shader = self._setup_shader(shading)
        self._renderer = MeshRenderer(self._rasterizer, self._shader)

    def __str__(self):
        """Returns the name of the renderer (nice for testing)."""
        return "pytorch3d"

    def _setup_settings(self, viewport):
        """Creates a ``RasterizationSettings`` object.
        
        Parameters
        ----------
        viewport : tuple[int]
            Desired output image size (width, height), in pixels
            
        Returns
        -------
        RasterizationSettings
            A pytorch3d ``RasterizationSettings`` object
        """
        if isinstance(viewport, (list, tuple, np.ndarray)):
            # pytorch3d wants (H, W), but we always use (W, H) to
            # keep the renderer API consistent
            viewport_ = (viewport[1], viewport[0])
        else:
            viewport_ = (viewport, viewport)

        return RasterizationSettings(
            image_size=(int(viewport_[0]), int(viewport_[1])),
            blur_radius=0.0,
            bin_size=0,
            faces_per_pixel=1,
        )

    def _setup_cameras(self, cam_mat, cam_type):
        """Sets of the appropriate camameras.
        
        Parameters
        ----------
        cam_mat : torch.tensor
            Camera matrix to be used
        cam_type : str
            Either 'orthographic' or 'perspective'
            
        Returns
        -------
        cam : pytorch3d camera
            A pytorch3d camera object (either ``FoVOrthographicCameras`` or
            ``FoVPerspectiveCameras``)
        """
        # In Medusa, we use the openGL convention of
        #   x-axis: left (-) to right (+)
        #   y-axis: bottom (-) to top (+)
        #   z-axis: forward (-) to backward (+),
        # but pytorch3d uses a flipped x and z-axis, to
        # we define a "flip matrix" to convert the camera matrix
        flip_mat = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        if cam_mat is None:
            cam_mat = np.eye(4)

        # For now, we'll do this in numpy; perhaps not very efficient
        # but I'm too lazy to implement this in pytorch and, assuming a
        # static camera, this only has to be executed once, anyway
        # (i.e., during initialization of the renderer)
        if torch.is_tensor(cam_mat):
            cam_mat = cam_mat.cpu().numpy().squeeze()

        # Convert column-major order (medusa/opengl) to row-major order (pytorch3d)
        # by transposing the camera matrix; then, flip it to pytorch3d coordinates
        cam_mat = flip_mat @ cam_mat.T
        R = torch.as_tensor(cam_mat[:3, :3], device=self.device).unsqueeze(0)
        T = torch.as_tensor(cam_mat[3, :3], device=self.device).unsqueeze(0)

        if cam_type == "orthographic":
            cam = FoVOrthographicCameras(
                device=self.device, T=T, R=R, znear=1.0, zfar=100.0
            )
        elif cam_type == "perspective":
            fov = 2 * np.arctan(self.viewport[1] / (2 * self.viewport[0]))
            cam = FoVPerspectiveCameras(
                fov=fov,
                znear=1.0,
                zfar=10000.0,
                degrees=False,
                device=self.device,
                R=R,
                T=T,
            )
        else:
            raise ValueError(f"Unknown camera type '{cam_type}'!")

        return cam

    def _setup_shader(self, shading):
        """Sets of the shader according to the ``shading`` parameter."""

        blend_params = BlendParams(background_color=(0, 0, 0))

        if shading == "flat":
            shader = HardFlatShader(
                self.device, self._cameras, self._lights, None, blend_params
            )
        elif shading == "smooth":
            shader = HardPhongShader(
                self.device, self._cameras, self._lights, None, blend_params
            )
        else:
            raise ValueError("For now, choose either 'flat' or 'smooth'!")

        return shader

    def __call__(self, v, tris, overlay=None, single_image=True):
        """Performs the actual rendering for a given (batch of) mesh(es).

        Parameters
        ----------
        v : torch.tensor
            A 3D (batch size x vertices x 3) tensor with vertices
        tris : torch.tensor
            A 3D (batch size x vertices x 3) tensor with triangles
        overlay : torch.tensor
            WIP
        cmap_name : str
            Name of (matplotlib) colormap; only relevant if ``overlay`` is not ``None``

        Returns
        -------
        img : torch.tensor
            A 4D tensor with uint8 values of shape batch size x ``viewport[0]`` x
            ``viewport[1]`` x 3 (RGB)
        """

        v, tris, overlay = self._preprocess(v, tris, overlay, format="torch")
        tris = tris.repeat(v.shape[0], 1, 1)

        if overlay is None:
            overlay = TexturesVertex(torch.ones_like(v, device=self.device))
        elif torch.is_tensor(overlay):
            overlay = TexturesVertex(overlay)

        meshes = Meshes(v, tris, overlay)
        imgs = self._renderer(meshes)

        if single_image:
            imgs = torch.amax(imgs, dim=0, keepdim=True)

        imgs = (imgs * 255).to(torch.uint8)

        return imgs

    def close(self):
        """Does nothing but is included to be consistent with the ``PyRenderer`` class."""
        pass

"""Module with a renderer class based on ``pytorch3d``."""

import torch
from torch import nn
import numpy as np

from pytorch3d.renderer import (FoVOrthographicCameras, FoVPerspectiveCameras,
                                HardFlatShader, HardPhongShader, MeshRendererWithFragments,
                                MeshRasterizer, MeshRenderer, PointLights,
                                RasterizationSettings, TexturesVertex, BlendParams)
from pytorch3d.structures import Meshes
from torchvision.utils import draw_keypoints, save_image

from ..defaults import DEVICE


class PytorchRenderer(nn.Module):
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
        Type of shading ('flat', 'smooth')
    wireframe_opts : None, dict
        Dictionary with extra options for wireframe rendering (options: 'width', 'color')
    device : str
        Device to store the image on ('cuda' or 'cpu')
    """
    def __init__(self, viewport, cam_mat, cam_type, shading="flat", lights=None,
                 background=(0, 0, 0), device=DEVICE):
        """Initializes a PytorchRenderer object."""
        super().__init__()
        self.viewport = viewport
        self.device = device
        self.cam_mat = cam_mat
        self.cam_type = cam_type
        self.background = background
        self._settings = self._setup_settings(viewport)
        self._cameras = self._setup_cameras(cam_mat, cam_type)
        self._lights = self._setup_lights(lights)
        self._rasterizer = MeshRasterizer(self._cameras, self._settings)
        self._shader = self._setup_shader(shading)
        self._renderer = MeshRenderer(self._rasterizer, self._shader)
        self.to(device).eval()

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
        flip_mat = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], device=self.device, dtype=torch.float32
        )

        if cam_mat is None:
            cam_mat = torch.eye(4, dtype=torch.float32, device=self.device)

        if isinstance(cam_mat, np.ndarray):
            cam_mat = torch.from_numpy(cam_mat).to(device=self.device, dtype=torch.float32)

        if cam_mat.device.type != self.device:
            cam_mat = cam_mat.to(device=self.device)

        # Convert column-major order (medusa/opengl) to row-major order (pytorch3d)
        # by transposing the camera matrix; then, flip it to pytorch3d coordinates
        #cam_mat[0, 0] *= -1
        #cam_mat[2, 2] *= -1
        cam_mat = flip_mat @ cam_mat.T

        # I'm so confused, but inverting the camera matrix again is necessary...
        cam_mat = torch.linalg.inv(cam_mat)

        R = cam_mat[:3, :3].unsqueeze(0)
        T = cam_mat[3, :3].unsqueeze(0)

        if cam_type == "orthographic":
            cam = FoVOrthographicCameras(
                device=self.device, T=T, R=R, znear=1.0, zfar=100.0
            )
        elif cam_type == "perspective":
            fov = 2 * np.arctan(self.viewport[1] / (2 * self.viewport[0]))
            cam = FoVPerspectiveCameras(
                fov=fov, znear=1., zfar=10_000.0, degrees=False, device=self.device,
                R=R, T=T
            )
        else:
            raise ValueError(f"Unknown camera type '{cam_type}'!")

        return cam

    def _setup_lights(self, lights):

        if lights is None:  # Pointlight is default
            # By default, we'll set the light to be at the same position as the camera
            cam_T = self._cameras.T

            # For orthographic cameras, we'll set the light to be 5 units away
            # (looks a bit better)
            if self.cam_type == 'orthographic':
                cam_T[:, 2] = 20
            return PointLights(device=self.device, location=cam_T)

        elif lights == 'sh':
            # Spherical harmonics, to be added later
            return 'sh'
        else:
            # Assume it's a Pytorch3d light class
            return lights

    def _setup_shader(self, shading):
        """Sets of the shader according to the ``shading`` parameter."""

        blend_params = BlendParams(background_color=self.background)

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

    def _create_meshes(self, v, tris, overlay):

        v, tris, overlay = self._preprocess(v, tris, overlay, format="torch")

        if overlay is None:
            # Note to self: Meshes *always* needs a texture, so create a dummy one
            overlay = TexturesVertex(torch.ones_like(v, device=self.device))
        elif torch.is_tensor(overlay):
            overlay = TexturesVertex(overlay)

        meshes = Meshes(v, tris, textures=overlay)

        return meshes

    def draw_landmarks(self, img, lms, radius=3, colors='green'):

        if img.shape[0] != 3:
            img = img.permute(2, 0, 1)

        lms = self._cameras.transform_points_screen(lms, image_size=img.shape[1:])
        return draw_keypoints(img, lms[:, :, :2], radius=radius, colors=colors)

    def _preprocess(self, v, tris, overlay=None, format="numpy"):
        """Performs some basic preprocessing of the vertices and triangles that
        is common to all renderers.

        Parameters
        ----------
        v : torch.tensor
            A B (optional) x V (vertices) x 3 tensor with vertices
        tris : torch/tensor
            A B (optional) x T (nr of triangles) x 3 (nr. of vertices per triangle) tensor
            with triangles
        format : str
            Either 'numpy' or 'torch'

        Returns
        -------
        v : torch.tensor
            Preprocessed vertices
        tris : torch.tensor
            Preprocessed triangles
        """
        if v.ndim == 2:
            v = v[None, ...]

        if tris.ndim == 2:
            tris = tris.repeat(v.shape[0], 1, 1)

        if torch.is_tensor(overlay):

            if overlay.ndim == 2:
                overlay = overlay.repeat(v.shape[0], 1, 1)

            if overlay.shape[-1] == 4:
                # remove alpha channel
                overlay = overlay[..., :3]

        if format == "numpy":
            if torch.is_tensor(v):
                v = v.cpu().numpy()

            if torch.is_tensor(tris):
                tris = tris.cpu().numpy()

            if torch.is_tensor(overlay):
                overlay = overlay.cpu().numpy()

        return v, tris, overlay

    def alpha_blend(self, img, background, face_alpha=None):
        """Simple alpha blend of a rendered image and a background. The image
        (`img`) is assumed to be an RGBA image and the background
        (`background`) is assumed to be a RGB image. The alpha channel of the
        image is used to blend them together. The optional `threshold`
        parameter can be used to impose a sharp cutoff.

        Parameters
        ----------
        img : torch.tensor
            A 3D or 4D tensor of shape (batch size) x height x width x 4 (RGBA)
        background : np.ndarray
            A 3D or 4D tensor shape height x width x 3 (RGB[A])

        Returns
        -------
        img : torch.tensor
            A blended image
        """

        alpha = img[..., 3, None] / 255.0

        if face_alpha is not None:
            alpha[alpha > face_alpha] = face_alpha

        if torch.is_tensor(img):
            if not torch.is_tensor(background):
                background = torch.as_tensor(background, device=img.device)

        if background.ndim == 3:
            # Assuming just a single image
            background = background[None, ...]

        if background.shape[-1] == 4:
            background = background[..., :3]

        if img.ndim == 3:
            img = img[None, ...]

        if background.shape[1] == 3:
            # channels_first -> channels_last
            background = background.permute(0, 2, 3, 1)

        # Fill black pixels in rendered image with background for nicer renderings,
        # especially with texturing
        img = img[..., :3] * alpha + (1 - alpha) * background
        idx = (img == 0).all(dim=-1)[..., None].repeat(1, 1, 1, 3)
        img[idx] = background[idx].float()

        if torch.is_tensor(img):
            img = img.to(dtype=torch.uint8)
        else:
            img = img.astype(np.uint8)

        # TODO: add global alpha level for face
        return img

    @staticmethod
    def save_image(f_out, img):
        """Saves a single image (using ``PIL``) to disk.

        Parameters
        ----------
        f_out : str, Path
            Path where the image should be saved
        """

        if img.ndim == 3:
            img = img.unsqueeze(0)

        if img.shape[3] in (3, 4):
            # channels_last -> channels_first
            img = img.permute(0, 3, 1, 2)

        if img.shape[1] == 4:
            img = img[:, :3, ...]

        save_image(img.float(), f_out, normalize=True)

    def forward(self, v, tris, overlay=None, single_image=True):
        """Performs the actual rendering for a given (batch of) mesh(es).

        Parameters
        ----------
        v : torch.tensor
            A 3D (batch size x vertices x 3) tensor with vertices
        tris : torch.tensor
            A 3D (batch size x vertices x 3) tensor with triangles
        overlay : torch.tensor
            A tensor with shape (batch size x vertices) with vertex colors
        single_image : bool
            Whether a single image with (potentially) multiple faces should be
            renderer (True) or multiple images with a single face should be renderered
            (False)

        Returns
        -------
        img : torch.tensor
            A 4D tensor with uint8 values of shape batch size x h x w x 3 (RGB), where
            h and w are defined in the viewport
        """

        meshes = self._create_meshes(v, tris, overlay)
        imgs = self._renderer(meshes)

        if single_image:
            imgs = torch.amax(imgs, dim=0, keepdim=True)

        imgs = (imgs * 255).to(torch.uint8)
        return imgs

"""Module with a renderer class based on ``pytorch3d``."""

from math import pi, sqrt
import numpy as np
import torch
from pytorch3d.renderer import (FoVOrthographicCameras, FoVPerspectiveCameras,
                                HardFlatShader, HardPhongShader,
                                MeshRasterizer, MeshRendererWithFragments, PointLights,
                                RasterizationSettings, TexturesVertex, BlendParams)
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes

from ...defaults import DEVICE
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
        Type of shading ('flat', 'smooth')
    wireframe_opts : None, dict
        Dictionary with extra options for wireframe rendering (options: 'width', 'color')
    device : str
        Device to store the image on ('cuda' or 'cpu')
    """
    def __init__(
        self,
        viewport,
        cam_mat,
        cam_type,
        shading="flat",
        lights=None,
        device=DEVICE,
    ):
        """Initializes a PytorchRenderer object."""
        self.viewport = viewport
        self.device = device
        self.cam_mat = cam_mat
        self.cam_type = cam_type
        self._settings = self._setup_settings(viewport)
        self._cameras = self._setup_cameras(cam_mat, cam_type)
        self._lights = self._setup_lights(lights)
        self._rasterizer = MeshRasterizer(self._cameras, self._settings)
        self._shader = self._setup_shader(shading)
        self._renderer = MeshRendererWithFragments(self._rasterizer, self._shader)
        self._fn = None

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

    def _create_meshes(self, v, tris, overlay):

        v, tris, overlay = self._preprocess(v, tris, overlay, format="torch")

        if overlay is None:
            # Note to self: Meshes *always* needs a texture, so create a dummy one
            overlay = TexturesVertex(torch.ones_like(v, device=self.device))
        elif torch.is_tensor(overlay):
            overlay = TexturesVertex(overlay)

        meshes = Meshes(v, tris, textures=overlay)

        return meshes

    def __call__(self, v, tris, overlay=None, single_image=True, return_frags=False):
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
        #self._fn = meshes.verts_normals_packed()[meshes.faces_packed()]
        imgs, frags = self._renderer(meshes)

        if single_image:
            imgs = torch.amax(imgs, dim=0, keepdim=True)

        imgs = (imgs * 255).to(torch.uint8)

        if return_frags:
            return imgs, frags
        else:
            return imgs

    def normal_map(self, fragments=None, v=None, tris=None):

        if fragments is None:
            meshes = self._create_meshes(v, tris, None, format="torch")
            fragments = self._rasterizer(
                meshes,
                raster_settings=self._settings,
                camers=self._cameras
            )

        #f = meshes.faces_packed()  # (F, 3)
        #vn = meshes.verts_normals_packed()  # (V, 3)

        # No need to interpolate the barycentric coordinates for normal map
        # (see https://github.com/facebookresearch/pytorch3d/issues/865)
        bcc = torch.ones_like(fragments.bary_coords, device=self.device)

        # N x H x W x K (1) x D
        normals = interpolate_face_attributes(fragments.pix_to_face, bcc, self._fn)
        normals = normals.squeeze(-2)  # get rid of 'K' dimension
        #normals = normals / torch.norm(normals, dim=-1, keepdim=True)
        #normals = (torch.nan_to_num(normals, nan=0.0) * 255.).to(torch.uint8)

        return normals

    def add_sh_light(self, img, fragments, sh_coeff):

        constant = torch.tensor(
            [
                1 / sqrt(4 * pi),
                ((2 * pi) / 3) * (sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (sqrt(3 / (4 * pi))),
                (pi / 4) * (3) * (sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (sqrt(5 / (12 * pi))),
                (pi / 4) * (3 / 2) * (sqrt(5 / (12 * pi))),
                (pi / 4) * (1 / 2) * (sqrt(5 / (4 * pi))),
            ], device=self.device
        )

        normal_imgs = self.normal_map(fragments)
        N = normal_imgs.permute(0, 3, 1, 2)
        sh = torch.stack([  # [bz, 9, h, w]
                N[:, 0]* 0. +1., N[:, 0], N[:, 1], \
                N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
                N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
            ], 1
        )

        sh = sh * constant[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1) # [bz, 9, 3, h, w]
        shading = shading.permute(0, 2, 3, 1)  # N x H x W x 3
        img[..., :3] = ((img[..., :3] * shading) * 1).to(torch.uint8)
        return img

    # def _world2uv(self, vt, tris_uv):

    #     meshes = Meshes(vt, tris_uv)
    #     fragments = rasterize_meshes(
    #         meshes,
    #         image_size=(256, 256),
    #         blur_radius=0.0,
    #         faces_per_pixel=1,
    #         bin_size=0,
    #         perspective_correct=False,
    #     )
    #     face_vertices = v[tris]


    # def extract_texture(self, imgs, v):

    #     v_screen = (self._cameras.transform_points_ndc(v, image_size=self.viewport) + 1 ) / 2
    #     self._rasterizer()
    #     self._renderer

    def close(self):
        pass

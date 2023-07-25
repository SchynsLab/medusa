"""Adapted from https://github.com/pomelyu/SHLight_pytorch, by Chien Chin-yu."""

import torch
from math import pi, sqrt
from pytorch3d.renderer.utils import TensorProperties, convert_to_tensors_and_broadcast

from medusa.defaults import DEVICE


class SphericalHarmonicsLights(TensorProperties):
    """An implementation of spherical harmonics lighting, adapted to work with
    the SH parameters returned by DECA/EMOCA.

    Parameters
    ----------
    sh_params : torch.Tensor
        Tensor of shape (B, 9, 3) containing SH parameters
    ambient_color : tuple
        Color of ambient light
    device : str
        Device to use (e.g., "cuda", "cpu")
    """
    def __init__(self, sh_params, ambient_color=(0.5, 0.5, 0.5), device=DEVICE):

        super().__init__(device=device,ambient_color=(ambient_color,),
                         sh_params=sh_params)

        sh_coeff = torch.tensor([
            1 / sqrt(4 * pi),
            ((2 * pi) / 3) * (sqrt(3 / (4 * pi))),
            ((2 * pi) / 3) * (sqrt(3 / (4 * pi))),
            ((2 * pi) / 3) * (sqrt(3 / (4 * pi))),
            (pi / 4) * (3) * (sqrt(5 / (12 * pi))),
            (pi / 4) * (3) * (sqrt(5 / (12 * pi))),
            (pi / 4) * (3) * (sqrt(5 / (12 * pi))),
            (pi / 4) * (3 / 2) * (sqrt(5 / (12 * pi))),
            (pi / 4) * (1 / 2) * (sqrt(5 / (4 * pi)))
        ], device=self.device)

        self.register_buffer("sh_coeff", sh_coeff[None, None, :])

    def clone(self):
        """Clones the object."""
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points) -> torch.Tensor:
        """Computes diffuse lighting."""
        # normals: (B, ..., 3)
        input_shape = normals.shape
        B = input_shape[0]
        normals = normals.view(B, -1, 3)
        # normals: (B, K, 3)

        sh = torch.stack([
            torch.ones_like(normals[..., 0]),               # 1
            normals[..., 0],                                # Y
            normals[..., 1],                                # Z
            normals[..., 2],                                # X
            normals[..., 1] * normals[..., 0],              # YX
            normals[..., 0] * normals[..., 2],              # YZ
            normals[..., 1] * normals[..., 2],
            normals[..., 0] ** 2 - normals[..., 1] ** 2,    # X^2 - Y^2
            3 * (normals[..., 2] ** 2) - 1,                 # 3Z^2 - 1
        ], dim=-1)

        sh, sh_coeff, sh_params = convert_to_tensors_and_broadcast(
            sh, self.sh_coeff, self.sh_params, device=normals.device
        )
        sh = sh * sh_coeff
        # sh_params: (B, 9, 3)
        color = torch.einsum("ijk,ikl->ijl", sh, sh_params)
        color = color.view(B, *input_shape[1:-1], 3)
        return color

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        """Computes specular lighting."""
        return self._zeros_channels(points)

    def _zeros_channels(self, points: torch.Tensor) -> torch.Tensor:
        ch = self.ambient_color.shape[-1]
        return torch.zeros(*points.shape[:-1], ch, device=points.device)

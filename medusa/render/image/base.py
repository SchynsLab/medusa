"""Module with a renderer base class."""
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image


class BaseRenderer(ABC):
    """A base class for the renderers in Medusa."""

    @abstractmethod
    def close(self):
        """Closes the currently used renderer."""
        pass

    def _preprocess(self, v, tris, tex, format="numpy"):
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

        if format == "numpy":
            if torch.is_tensor(v):
                v = v.cpu().numpy()

            if torch.is_tensor(tris):
                tris = tris.cpu().numpy()

            if torch.is_tensor(tex):
                tex = tex.cpu().numpy()

        return v, tris, tex

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

        if torch.is_tensor(img):
            img = img.cpu().numpy()

        if img.ndim == 4:
            if img.shape[0] != 1:
                raise ValueError("Cannot save batch of images")

            img = img[0, ...]

        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        Image.fromarray(img).save(str(f_out))

    @staticmethod
    def load_image(f_in, device=None):
        """Utility function to read a single image to disk (using ``PIL``).

        Parameters
        ----------
        f_in : str, Path
            Path of file
        device : None, str
            If ``None``, the image is returned as a numpy array; if 'cuda' or 'cpu',
            the image is returned as a torch tensor
        """
        img = np.array(Image.open(str(f_in)))

        if device is not None:
            img = torch.as_tensor(img, device=device)

        return img

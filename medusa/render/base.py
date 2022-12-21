import cv2
import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseRenderer(ABC):
    @abstractmethod
    def close(self):
        pass

    def _preprocess(self, v, tris, format="numpy"):

        if v.ndim == 2:
            v = v[None, ...]

        if format == "numpy":
            if torch.is_tensor(v):
                v = v.cpu().numpy()

            if torch.is_tensor(tris):
                tris = tris.cpu().numpy()

        return v, tris

    def alpha_blend(self, img, background, face_alpha=None):
        """Simple alpha blend of a rendered image and a background. The image
        (`img`) is assumed to be an RGBA image and the background
        (`background`) is assumed to be a RGB image. The alpha channel of the
        image is used to blend them together. The optional `threshold`
        parameter can be used to impose a sharp cutoff.

        Parameters
        ----------
        img : np.ndarray
            A 3D numpy array of shape height x width x 4 (RGBA)
        background : np.ndarray
            A 3D numpy array of shape height x width x 3 (RGB)
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

        img = img[..., :3] * alpha + (1 - alpha) * background

        if torch.is_tensor(img):
            img = img.to(dtype=torch.uint8)
        else:
            img = img.astype(np.uint8)

        # TODO: add global alpha level for face
        return img

    @staticmethod
    def save_image(f_out, img):

        if torch.is_tensor(img):
            img = img.cpu().numpy()

        if img.ndim == 4:
            if img.shape[0] != 1:
                raise ValueError("Cannot save batch of images")

            img = img[0, ...]

        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        cv2.imwrite(str(f_out), img[:, :, [2, 1, 0]])

    @staticmethod
    def load_image(f_in, device=None):

        img = cv2.imread(str(f_in))
        img = img[:, :, [2, 1, 0]]

        if device is not None:
            img = torch.as_tensor(img, device=device)

        return img

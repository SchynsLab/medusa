import pickle
from pathlib import Path

import torch
import yaml

from ..base import BaseReconModel
from ...data import get_template_flame


class FlameReconModel(BaseReconModel):

    def _preprocess(self, imgs):

        if imgs.dtype == torch.uint8:
            imgs = imgs.float()

        if imgs.max() >= 1.0:
            imgs = imgs.div_(255.0)

        return imgs

    def is_dense(self):

        dense = False
        if hasattr(self, "name"):
            if "dense" in self.name:
                dense = True

        return dense

    def get_tris(self):
        """Retrieves the triangles (tris) associated with the predicted vertex
        mesh; does lazy loading of the triangles from disk and, if there's an
        "active mask" (i.e., `apply_mask` has been called), the triangles will
        be indexed accordingly.

        Note that this implementation is very ugly and slow, but is
        probably only called once.
        """

        if not hasattr(self, "_tris"):
            topo = 'dense' if self.is_dense() else 'coarse'
            template = get_template_flame(topo, keys=['tris'], device=self.device)
            self._tris = template['tris']

        if hasattr(self, "_active_mask") and self._tris.shape[0] == 9976:
            idx = torch.isin(self._tris, self._active_mask).all(dim=1)
            lut = {k.item(): i for i, k in enumerate(self._active_mask)}
            self._tris = torch.as_tensor(
                [lut[x.item()] for x in self._tris[idx].flatten()],
                dtype=torch.int64,
                device=self.device,
            )
            self._tris = self._tris.reshape((idx.sum(), -1))

        return self._tris

    def apply_mask(self, name, v):

        if self.is_dense():
            raise ValueError("Cannot apply mask for dense reconstructions!")

        if not hasattr(self, "_masks"):
            with open(self.cfg["flame_masks_path"], "rb") as f_in:
                self._masks = pickle.load(f_in, encoding="latin1")

        if name in self._masks:
            self._active_mask = torch.as_tensor(
                self._masks[name], dtype=torch.int64, device=self.device
            )
        else:
            raise ValueError(f"Mask name '{name}' not in masks")

        return v[:, self._active_mask, :]

    def get_cam_mat(self):

        cam_mat = torch.eye(4, device=self.device)
        cam_mat[2, 3] = 4
        return cam_mat

    def close(self):

        if getattr(self, "crop_mat", None) is not None:
            self.crop_mat = None

        if getattr(self, "_tris", None) is not None:
            self._tris = None

        if getattr(self, "_active_mask", None) is not None:
            self._active_mask = None

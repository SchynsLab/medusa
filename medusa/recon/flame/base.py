from pathlib import Path

import torch
import yaml

from ..base import BaseReconModel
from .data import get_template_flame


class FlameReconModel(BaseReconModel):

    def _load_cfg(self):
        """Loads a (default) config file."""
        data_dir = Path(__file__).parents[2] / 'data/flame'
        cfg = data_dir / 'config.yaml'

        if not cfg.is_file():
            raise ValueError(f"Could not find {str(cfg)}!")

        with open(cfg, "r") as f_in:
            self.cfg = yaml.safe_load(f_in)

    def _preprocess(self, imgs):

        if imgs.dtype == torch.uint8:
            imgs = imgs.float()

        if imgs.max() >= 1.:
            imgs = imgs.div_(255.)

        return imgs

    def get_tris(self):
        """Retrieves the triangles (tris) associated with the predicted vertex
        mesh."""

        dense = False
        if hasattr(self, 'name'):
            if 'dense' in self.name:
                dense = True

        template = get_template_flame(dense=dense)
        return template['f']

    def close(self):

        if hasattr(self, 'crop_mat'):
            self.crop_mat = None

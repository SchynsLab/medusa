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

    def get_tris(self):
        """Retrieves the triangles (tris) associated with the predicted vertex
        mesh."""
        if hasattr(self, 'dense'):
            dense = self.dense
        else:
            # Assume that we're using the coarse version (e.g., for MICA)
            dense = False

        template = get_template_flame(dense=dense)
        return template['f']

    def close(self):

        if hasattr(self, 'crop_mat'):
            self.crop_mat = None

"""Module with a base class for FLAME-based reconstruction models."""
import torch

from ..base import BaseReconModel


class FlameReconModel(BaseReconModel):
    """A reconstruction model which outputs data based on the FLAME-
    topology."""

    def _preprocess(self, imgs):
        """Does some basic preprocessing to the inputs.

        Parameters
        ----------
        imgs : torch.tensor
            A batch size x 3 x h x w tensor, either normalized or not

        Returns
        -------
        imgs : torch.tensor
            The normalized tensor in the right format
        """

        if imgs.dtype == torch.uint8:
            imgs = imgs.float()

        if imgs.max() >= 1.0:
            imgs = imgs.div_(255.0)

        return imgs

    def is_dense(self):
        """Checks if the current model is a dense model.

        Returns
        -------
        dense : bool
            True if dense, False otherwise
        """
        dense = False
        if hasattr(self, "name"):
            if "dense" in self.name:
                dense = True

        return dense

    def get_tris(self):
        """Retrieves the triangles (tris) associated with the predicted vertex
        mesh."""
        # Avoids circular import
        from ...data import get_template_flame

        topo = 'dense' if self.is_dense() else 'coarse'
        template = get_template_flame(topo, keys=['tris'], device=self.device)
        return template['tris']

    def get_cam_mat(self):
        """Returns a default camera matrix for FLAME-based reconstructions."""
        cam_mat = torch.eye(4, device=self.device)
        cam_mat[2, 3] = 4
        return cam_mat

    def close(self):
        """Sets loaded triangles to None."""
        if getattr(self, "_tris", None) is not None:
            self._tris = None

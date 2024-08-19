import torch
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

from ..geometry import compute_vertex_normals


class Overlay:
    """Class for creating color "overlays" to be rendered as vertex colors of a mesh.

    Parameters
    ----------
    v : torch.Tensor
        The values to be mapped to colors
    vmin : float, optional
        The minimum value of the colormap (default: minimum of v)
    vmax : float, optional
        The maximum value of the colormap (default: maximum of v)
    vcenter : float, optional
        The center value of the colormap (default: mean of v)
    dim : int or str
        If int (0, 1, 2), dimension to be visualized; if 'normals', the values are
        projected on the vertex normals
    v0 : torch.Tensor, optional
        If dim='normals', v0 represents the mesh from which the normals are computed
    tris : torch.Tensor, optional
        If dim='normals', tris represents the triangles from which the normals are computed
    norm : matplotlib.colors.Normalize, optional
        Normalization class (default: `TwoSlopeNorm`)
    """
    def __init__(self, v, cmap='bwr', vmin=None, vmax=None, vcenter=None,
                 dim='normals', v0=None, tris=None, norm=TwoSlopeNorm):

        self.v = v
        self.vmin = vmin
        self.vmax = vmax
        self.vcenter = vcenter
        self.dim = dim
        self.v0 = v0
        self.tris = tris
        self._norm = self._init_norm(norm)
        self._cmap = cm.get_cmap(cmap)

    def _init_norm(self, Norm):
        """Initializes the normalization object."""
        if self.vmin is None:
            self.vmin = self.v.min().item()

        if self.vmax is None:
            self.vmax = self.v.max().item()

        if self.vcenter is None:
            self.vcenter = self.v.mean().item()

        return Norm(vmin=self.vmin, vcenter=self.vcenter, vmax=self.vmax)

    def to_rgb(self):
        """Returns the RGB colors for the overlay.

        Returns
        -------
        colors : torch.Tensor
            RGB colors (N x V x 3)
        """
        v = self.v
        device = v.device

        if v.shape[-1] == 3:
            if self.dim == 'normals':
                # We're dealing with XYZ coordinates; project onto normal
                normals = compute_vertex_normals(self.v0, self.tris)
                v = (v * normals).sum(dim=-1)
            elif self.dim in (0, 1, 2):
                v = v[..., self.dim]
            else:
                raise ValueError(f'Invalid dim: {self.dim}, choose from (0, 1, 2, "normals")')
        else:
            # Assume XYZ dimension has been reduced already
            pass

        v = v.cpu().numpy()
        colors = self._cmap(self._norm(v))
        colors = torch.as_tensor(colors, device=device, dtype=torch.float32)

        return colors

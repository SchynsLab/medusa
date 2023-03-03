import torch
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

from ..geometry import compute_vertex_normals
from ..data import get_tris


class Overlay:

    def __init__(self, v, colormap='bwr', vmin=None, vmax=None, vcenter=0):
        self.v = v
        self.vmin = vmin
        self.vmax = vmax
        self.vcenter = vcenter
        self._cmap = cm.get_cmap(colormap)

    def _create_tsn(self, v):

        if self.vmin is None:
            self.vmin = v.min()

        if self.vmax is None:
            self.vmax = v.max()

        return TwoSlopeNorm(vmin=self.vmin, vcenter=self.vcenter, vmax=self.vmax)

    def to_array(self, v0=None, dim='normals', alpha=False, tris=None):

        v = self.v
        device = v.device

        if v.shape[-1] == 3:
            if dim == 'normals':
                # We're dealing with XYZ coordinates; project onto normal
                if tris is None:
                    tris = get_tris(topo='flame-coarse', device=device)

                normals = compute_vertex_normals(v0, tris)
                v = (v * normals).sum(dim=-1)
            elif dim in (0, 1, 2):
                v = v[..., dim]
            else:
                raise ValueError(f'Invalid dim: {dim}, choose from (0, 1, 2, "normals")')
        else:
            # Assume XYZ dimension has been reduced already
            pass

        # TODO: not only two-slope norm
        tsn = self._create_tsn(v)

        v = v.cpu().numpy()
        v = self._cmap(tsn(v))

        if device is not None:
            v = torch.as_tensor(v, device=device, dtype=torch.float32)

        if not alpha:
            v = v[..., :3]

        return v

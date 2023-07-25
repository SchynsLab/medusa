import torch

from ..transforms import estimate_similarity_transform
from ..defaults import DEVICE
from ..data import get_rigid_vertices, get_vertex_template


def estimate_alignment(v, topo, target=None, estimate_scale=False, device=DEVICE):
    """Aligment of a temporal series of 3D meshes to a target (which should
    have the same topology).

    Parameters
    ----------
    v : torch.tensor
        A float tensor with vertices of shape B (batch size) x V (vertices) x 3
    topo : str
        Topology corresponding to ``v``
    target : torch.tensor
        Target to use for alignment; if ``None`` (default), a default template will be
        used
    estimate_scale : bool
        Whether the alignment may also involve scaling
    device : str
        Either 'cuda' (GPU) or 'cpu'

    Returns
    -------
    mat : torch.tensor
        A float tensor with affine matrices of shape B (batch size) x 4 x 4
    """

    v_idx = get_rigid_vertices(topo, device)

    if target is None:
        target = get_vertex_template(topo, device)

    mat = estimate_similarity_transform(v[:, v_idx, :], target[v_idx], estimate_scale)
    mat = torch.linalg.inv(mat)

    return mat

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


# def apply_alignment(v, mat, cam_mat, reference_index=0):
#     """Applies the estimated """
#     v = transform_points(torch.linalg.inv(mat), v)
#     # Because the object space changed (world --> local) and we assume
#     # the object will be visualized on top of the video frames, we need
#     # to update the camera matrix with the world-to-local matrix; this ensure
#     # the rendering will be in the "right" pixel space; note that we only do
#     # this for the first frame (otherwise we 're-introduce' the global motion,
#     # but this time in the camera, and we assume a fixed camera of course)
#     cam_mat = torch.linalg.inv(mat[reference_index]) @ cam_mat
#     # Fix floating point inaccuracy
#     cam_mat[3, :] = torch.tensor([0., 0., 0., 1.], device=cam_mat.device)

#     return v, cam_mat

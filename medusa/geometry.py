"""Module with geometry-related functionality."""
import torch


def compute_tri_normals(v, tris, normalize=True):
    """Computes triangle (surface/face) normals.
    
    Parameters
    ----------
    v : torch.tensor
        A float tensor with vertices of shape B (batch size) x V (vertices) x 3
    tris : torch.tensor
        A long tensor with indices of shape T (triangles) x 3 (vertices per triangle)
    normalize : bool
        Whether to normalize the normals (usually, you want to do this, but included
        here so it can be reused when computing vertex normals, which uses
        unnormalized triangle normals)
    
    Returns
    -------
    fn : torch.tensor
        A float tensor with triangle normals of shape B (batch size) x T (triangles) x 3
    """
    vf = v[:, tris]
    fn = torch.cross(vf[:, :, 2] - vf[:, :, 1], vf[:, :, 0] - vf[:, :, 1], dim=2)

    if normalize:
        # To be consistent with pytorch3d, set minimum of norm to 1e-6
        norm = torch.linalg.norm(fn, dim=2, keepdim=True)
        norm = torch.clamp_min(norm, 1e-6)
        fn = fn / norm

    return fn


def compute_vertex_normals(v, tris):
    """Computes vertex normals in a vectorized way, based on the ``pytorch3d``
    implementation.
    
    Parameters
    ----------
    v : torch.tensor
        A float tensor with vertices of shape B (batch size) x V (vertices) x 3
    tris : torch.tensor
        A long tensor with indices of shape T (triangles) x 3 (vertices per triangle)
    
    Returns
    -------
    vn : torch.tensor
        A float tensor with vertex normals of shape B (batch size) x V (vertices) x 3
    """

    device = v.device
    vn = torch.zeros_like(v, device=device)

    fn = compute_tri_normals(v, tris, normalize=False)
    vn.index_add_(1, tris[:, 0], fn)
    vn.index_add_(1, tris[:, 1], fn)
    vn.index_add_(1, tris[:, 2], fn)
    vn = torch.nn.functional.normalize(vn, eps=1e-6, dim=2)
    
    return vn

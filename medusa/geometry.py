"""Module with geometry-related functionality.

For now only contains functions to compute vertex and triangle normals.
"""

import torch
import pickle
from .data import get_external_data_config


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

    if v.ndim == 2:
        v = v.unsqueeze(0)

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

    if v.ndim == 2:
        v = v.unsqueeze(0)

    vn = torch.zeros_like(v, device=v.device)

    fn = compute_tri_normals(v, tris, normalize=False)
    vn.index_add_(1, tris[:, 0], fn)
    vn.index_add_(1, tris[:, 1], fn)
    vn.index_add_(1, tris[:, 2], fn)
    vn = torch.nn.functional.normalize(vn, eps=1e-6, dim=2)

    return vn


def apply_vertex_mask(name, **attrs):
    """Applies a vertex mask to a tensor of vertices.

    Parameters
    ----------
    v : torch.tensor
        A float tensor with vertices of shape B (batch size) x V (vertices) x 3
    name : str
        Name of mask to apply

    Returns
    -------
    v_masked : torch.tensor
        A float tensor with masked vertices of shape B (batch size) x V (vertices) x 3
    """

    if not attrs:
        raise ValueError("No attributes to apply mask to!")

    masks_path = get_external_data_config(key='flame_masks_path')
    with open(masks_path, "rb") as f_in:
        masks = pickle.load(f_in, encoding="latin1")
        if name not in masks:
            raise ValueError(f"Mask name '{name}' not in masks")

    device = attrs[list(attrs.keys())[0]].device
    mask = torch.as_tensor(masks[name], dtype=torch.int64, device=device)

    if 'v' in attrs:
        attrs['v'] = attrs['v'][:, mask, :]

    if 'vt' in attrs:
        attrs['vt'] = attrs['vt'][..., mask, :]

    if 'tris' in attrs:
        # This is ugly/slow, but create a look-up table mapping mask values to new
        # indices (from 0, 1, ... len(mask))
        lut = {k.item(): i for i, k in enumerate(mask)}

        # We also need to filter out the triangles that contain vertices that are not
        # part of the mask! First, find which triangles contain only vertices part of
        # the mask
        idx = torch.isin(attrs['tris'], mask).all(dim=1)
        # Finally, map old indices to new indices and unflatten
        attrs['tris'] = torch.as_tensor([lut[x.item()] for x in attrs['tris'][idx].flatten()],
                                        dtype=torch.int64, device=device)
        attrs['tris'] = attrs['tris'].reshape((idx.sum(), -1))

        #if 'tris_uv' in attrs:
        #    attrs['tris_uv'] = torch.as_tensor([lut[x.item()] for x in attrs['tris_uv'][idx].flatten()],
        #                                    dtype=torch.int64, device=device)
        #    attrs['tris_uv'] = attrs['tris_uv'].reshape((idx.sum(), -1))

    return attrs

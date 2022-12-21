import numpy as np
import torch
import torch.nn.functional as F


def face_vertices(v, f):
    bs, nv = v.shape[:2]
    bs = f.shape[0]
    device = v.device
    f = f + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    v = v.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return v[f.long()]


def vertex_normals(v, f):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """

    bs, nv = v.shape[:2]
    bs, _ = f.shape[:2]
    device = v.device
    normals = torch.zeros(bs * nv, 3).to(device)

    f = (
        f + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    )  # expanded faces
    vf = v.reshape((bs * nv, 3))[f.long()]

    f = f.reshape(-1, 3).long()
    vf = vf.reshape(-1, 3, 3)

    normals.index_add_(
        0, f[:, 1], torch.cross(vf[:, 2] - vf[:, 1], vf[:, 0] - vf[:, 1])
    )
    normals.index_add_(
        0, f[:, 2], torch.cross(vf[:, 0] - vf[:, 2], vf[:, 1] - vf[:, 2])
    )
    normals.index_add_(
        0, f[:, 0], torch.cross(vf[:, 1] - vf[:, 0], vf[:, 2] - vf[:, 0])
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    return normals


def upsample_mesh(v, normals, disp_map, dense_template):
    x_coords = dense_template["x_coords"]
    y_coords = dense_template["y_coords"]
    valid_pixel_ids = dense_template["valid_pixel_ids"]
    valid_pixel_3d_faces = dense_template["valid_pixel_3d_faces"]
    valid_pixel_b_coords = dense_template["valid_pixel_b_coords"]

    pixel_3d_points = (
        v[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis]
        + v[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis]
        + v[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    )

    pixel_3d_normals = (
        normals[valid_pixel_3d_faces[:, 0], :]
        * valid_pixel_b_coords[:, 0][:, np.newaxis]
        + normals[valid_pixel_3d_faces[:, 1], :]
        * valid_pixel_b_coords[:, 1][:, np.newaxis]
        + normals[valid_pixel_3d_faces[:, 2], :]
        * valid_pixel_b_coords[:, 2][:, np.newaxis]
    )

    pixel_3d_normals = (
        pixel_3d_normals / np.linalg.norm(pixel_3d_normals, axis=-1)[:, np.newaxis]
    )
    displacements = disp_map[
        y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)
    ]
    offsets = np.einsum("i,ij->ij", displacements, pixel_3d_normals)
    v_dense = pixel_3d_points + offsets

    return v_dense

# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import cv2
import torch
import numpy as np
import torch.nn.functional as F


def upsample_mesh(vertices, normals, faces, displacement_map, texture_map, dense_template):
    ''' Credit to Timo
    upsampling coarse mesh (with displacment map)
        vertices: vertices of coarse mesh, [nv, 3]
        normals: vertex normals, [nv, 3]
        faces: faces of coarse mesh, [nf, 3]
        texture_map: texture map, [256, 256, 3]
        displacement_map: displacment map, [256, 256]
        dense_template: 
    Returns: 
        dense_vertices: upsampled vertices with details, [number of dense vertices, 3]
        dense_colors: vertex color, [number of dense vertices, 3]
        dense_faces: [number of dense faces, 3]
    '''
    dense_faces = dense_template['f']
    x_coords = dense_template['x_coords']
    y_coords = dense_template['y_coords']
    valid_pixel_ids = dense_template['valid_pixel_ids']
    valid_pixel_3d_faces = dense_template['valid_pixel_3d_faces']
    valid_pixel_b_coords = dense_template['valid_pixel_b_coords']

    pixel_3d_points = vertices[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                    vertices[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                    vertices[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    vertex_normals = normals
    pixel_3d_normals = vertex_normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                    vertex_normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                    vertex_normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    pixel_3d_normals = pixel_3d_normals / np.linalg.norm(pixel_3d_normals, axis=-1)[:, np.newaxis]
    displacements = displacement_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    dense_colors = texture_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    offsets = np.einsum('i,ij->ij', displacements, pixel_3d_normals)
    dense_vertices = pixel_3d_points + offsets
    return dense_vertices, dense_colors, dense_faces


def plot_kpts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1]==4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)

        if i in end_list:
            continue

        ed = kpts[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), radius)
        image = cv2.circle(image,(int(st[0]), int(st[1])), radius, c, radius*2)  

    return image


## load obj,  similar to load_obj from pytorch3d
def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = torch.tensor(verts, dtype=torch.float32)
    uvcoords = torch.tensor(uvcoords, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long); faces = faces.reshape(-1, 3) - 1
    uv_faces = torch.tensor(uv_faces, dtype=torch.long); uv_faces = uv_faces.reshape(-1, 3) - 1
    return (
        verts,
        uvcoords,
        faces,
        uv_faces
    )

# ---------------------------- process/generate vertices, normals, faces
def generate_triangles(h, w, margin_x=2, margin_y=5, mask = None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    #.
    # w*h
    triangles = []
    for x in range(margin_x, w-1-margin_x):
        for y in range(margin_y, h-1-margin_y):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:,[0,2,1]]
    return triangles


# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
    

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


def tensor2image(tensor, bgr=True):
    image = tensor.detach().cpu().numpy()
    image = image*255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)
    if bgr:
        image = image[:,:,[2,1,0]]

    return image.astype(np.uint8).copy()


def tensor_vis_landmarks(images, landmarks, gt_landmarks=None, color = 'g', isScale=True):
    # visualize landmarks
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1,2,0)[:,:,[2,1,0]].copy(); image = (image*255)
        if isScale:
            predicted_landmark = predicted_landmarks[i]
            predicted_landmark[...,0] = predicted_landmark[...,0]*image.shape[1]/2 + image.shape[1]/2
            predicted_landmark[...,1] = predicted_landmark[...,1]*image.shape[0]/2 + image.shape[0]/2
        else:
            predicted_landmark = predicted_landmarks[i]
        if predicted_landmark.shape[0] == 68:
            image_landmarks = plot_kpts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks, gt_landmarks_np[i]*image.shape[0]/2 + image.shape[0]/2, 'r')
        else:
            image_landmarks = plot_verts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks, gt_landmarks_np[i]*image.shape[0]/2 + image.shape[0]/2, 'r')
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(vis_landmarks[:,:,:,[2,1,0]].transpose(0,3,1,2))/255.#, dtype=torch.float32)
    return vis_landmarks

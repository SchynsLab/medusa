import torch
import pytest
from conftest import _is_pytorch3d_installed
from medusa.geometry import compute_tri_normals, compute_vertex_normals
from medusa.data import get_template_flame
from medusa.defaults import DEVICE


@pytest.mark.parametrize('batch_size', [1, 100])
def test_compute_tri_normals(batch_size):

    if not _is_pytorch3d_installed():
        return 
    else:
        from pytorch3d.structures import Meshes
    
    template = get_template_flame('coarse', keys=['v', 'tris'], device=DEVICE)
    v = template['v'].repeat((batch_size, 1, 1))
    tris = template['tris']

    fn = compute_tri_normals(v, tris, normalize=True)
    meshes = Meshes(v, tris.repeat((batch_size, 1, 1)))
    meshes._compute_face_areas_normals()
    fn_ = meshes._faces_normals_packed.reshape((batch_size, -1, 3))
    torch.testing.assert_close(fn, fn_)


@pytest.mark.parametrize('batch_size', [1, 100])
def test_compute_vertex_normals(batch_size):

    if not _is_pytorch3d_installed():
        return 
    else:
        from pytorch3d.structures import Meshes
    
    template = get_template_flame('coarse', keys=['v', 'tris'], device=DEVICE)
    v = template['v'].repeat((batch_size, 1, 1))
    tris = template['tris']

    vn = compute_vertex_normals(v, tris)
    meshes = Meshes(v, tris.repeat((batch_size, 1, 1)))
    meshes._compute_vertex_normals()
    vn_ = meshes._verts_normals_packed.reshape((batch_size, -1, 3))
    torch.testing.assert_close(vn, vn_)

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from conftest import _is_gha_compatible

from medusa.defaults import DEVICE
from medusa.containers import Data4D
from medusa.data import get_example_data4d, get_example_video


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_init(device):
    """Simple initialization test."""

    if not _is_gha_compatible(device):
        return

    v = torch.randn((100, 5023, 3), device=device)
    mat = torch.randn((100, 4, 4), device=device)
    tris = torch.randint(0, 100, (9000, 3), device=device)
    img_idx = torch.arange(100, device=device)
    face_idx = torch.zeros(100, device=device)
    metadata = {"img_size": (512, 512), "n_img": 100, "fps": 24}
    data = Data4D(v, mat, tris, img_idx, face_idx, metadata, device=device)


@pytest.mark.parametrize("kwargs", [{}, {'recon_model': 'emoca-coarse'}])
def test_from_video(kwargs):
    video_test = get_example_video()
    data = Data4D.from_video(video_test, **kwargs)
    if kwargs:
        assert(data.v.shape[1:] == (5023, 3))


def test_apply_mask():
    data = get_example_data4d(load=True, model='emoca-coarse')
    data.apply_vertex_mask('face')
    assert(data.v.shape[1] == 1787)
    assert(data.tris.shape[0] == 3408)


@pytest.mark.parametrize("model", ["mediapipe", "emoca-coarse"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_load_and_save(model, device):
    """Tests loading from disk and saving a Data4D object."""

    if not _is_gha_compatible(device):
        return

    h5 = get_example_data4d(load=False, model=model)
    data = Data4D.load(h5, device=device)
    assert isinstance(data, Data4D)

    with tempfile.NamedTemporaryFile() as f_out:
        data.save(f_out.name)
        assert Path(f_out.name).is_file()


@pytest.mark.parametrize("model", ["mediapipe", "emoca-coarse"])
def test_project68(model):
    """Tests projection of vertices onto a subset of 68 canonical vertices."""
    # Need to specify DEVICE here because otherwise Github Action tests error
    data = get_example_data4d(load=True, model=model, device=DEVICE)
    v68 = data.project_to_68_landmarks()
    assert v68.shape == (data.v.shape[0], 68, 3)


@pytest.mark.parametrize("model", ["mediapipe", "emoca-coarse"])
def test_to_local_and_to_world(model):
    data = get_example_data4d(load=True, model=model, device=DEVICE)
    data.to_local()
    assert(data.space == 'local')
    data.to_world()
    assert(data.space == 'world')


@pytest.mark.parametrize("pad_missing", [True, False])
@pytest.mark.parametrize("n_faces", [1, 3])
def test_get_face(pad_missing, n_faces):
    """Tests the extraction of faces from a Data4D object which has multiple
    faces."""

    data = get_example_data4d(n_faces, load=True, model='emoca-coarse')
    for index in range(n_faces):
        d = data.get_face(index, pad_missing)

        # Check if ``d`` is actually a Data4D object
        assert isinstance(d, Data4D)

        # Check if ``d`` only has face ID ``index``
        assert (d.face_idx == index).all()

        if pad_missing:
            assert d.v.shape[0] == d.mat.shape[0] == data.video_metadata["n_img"]
        else:
            n_ = (data.face_idx == index).sum()
            assert d.v.shape[0] == d.mat.shape[0] == n_

    with pytest.raises(ValueError):
        data.get_face(100)


@pytest.mark.parametrize("n_faces", [1, 3])
def test_decompose_mats(n_faces):
    """Tests decomposition of affine matrices into affine parameters."""

    data = get_example_data4d(n_faces, load=True, model='mediapipe')
    dfs = data.decompose_mats(to_df=True)

    if not isinstance(dfs, list):
        dfs = [dfs]

    for i, df in enumerate(dfs):
        f_out = f"./tests/test_viz/misc/{n_faces}_face_id-{i}.tsv"
        df.to_csv(f_out, sep="\t", index=False)

        fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(12, 6))
        for i, param in enumerate(["trans", "rot", "scale", "shear"]):
            axes[i].axhline(0, ls="--", c="k")
            data = df.iloc[:, i * 3 : (i + 1) * 3]
            data = data - data.dropna().iloc[0]
            axes[i].plot(data)
            axes[i].set_ylabel(param, fontsize=15)
            axes[i].set_xlim(0, df.shape[0])
            axes[i].spines.right.set_visible(False)
            axes[i].spines.top.set_visible(False)

        axes[i].set_xlabel("Frame nr.", fontsize=15)
        axes[i].legend(["still", "X", "Y", "Z"], frameon=False, ncol=3, fontsize=15)
        fig.savefig(f_out.replace(".tsv", ".png"))

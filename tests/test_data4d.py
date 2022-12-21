import torch
import pytest
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from medusa.containers import Data4D
from medusa.data import get_example_h5
from medusa.recon import videorecon

# @pytest.mark.parametrize('device', ['cpu', 'cuda'])
# def test_init(device):

#     v = torch.randn((100, 5023, 3), device=device)
#     mat = torch.randn((100, 4, 4), device=device)
#     tris = torch.randint(0, 100, (9000, 3), device=device)
#     img_idx = torch.arange(100, device=device)
#     face_idx = torch.zeros(100, device=device)
#     metadata = {
#         'img_size': (512, 512),
#         'n_img': 100,
#         'fps': 24
#     }
#     data = Data4D(v, mat, tris, img_idx, face_idx, metadata, device=device)


# @pytest.mark.parametrize('model', ['mediapipe', 'emoca-coarse'])
# @pytest.mark.parametrize('device', ['cpu', 'cuda'])
# def test_load_and_save(model, device):

#     h5 = get_example_h5(load=False, model=model)
#     data = Data4D.load(h5, device=device)

#     with tempfile.NamedTemporaryFile() as f_out:
#         data.save(f_out.name)
#         assert(Path(f_out.name).is_file())


# @pytest.mark.parametrize('model', ['mediapipe', 'emoca-coarse'])
# def test_project68(model):

#     data = get_example_h5(load=True, model=model)
#     v68 = data.project_to_68_landmarks()
#     assert(v68.shape[1:] == (68, 3))

# @pytest.mark.parametrize('pad_missing', [True, False])
# @pytest.mark.parametrize('video_test', [1, 3], indirect=True)
# def test_get_face(pad_missing, video_test):
#     data = videorecon(video_test, 'emoca-coarse')

#     n_exp = int(video_test.stem[0])
#     for index in range(n_exp):
#         d = data.get_face(index, pad_missing)

#         # Check if ``d`` is actually a Data4D object
#         assert(isinstance(d, Data4D))

#         # Check if ``d`` only has face ID ``index``
#         assert((d.face_idx == index).all())

#         if pad_missing:
#             assert(d.v.shape[0] == d.mat.shape[0] == data.video_metadata['n_img'])
#         else:
#             n_ = (data.face_idx == index).sum()
#             assert(d.v.shape[0] == d.mat.shape[0] == n_)

#     with pytest.raises(ValueError):
#         data.get_face(100)


@pytest.mark.parametrize("video_test", [1, 3], indirect=True)
def test_decompose_mats(video_test):
    data = videorecon(video_test, recon_model="mediapipe")
    dfs = data.decompose_mats(to_df=True)

    if not isinstance(dfs, list):
        dfs = [dfs]

    for i, df in enumerate(dfs):
        f_out = f"./tests/test_viz/misc/{video_test.stem}_id-{i}.tsv"
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

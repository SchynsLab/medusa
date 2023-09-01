import numpy as np
import pytest
import torch
from conftest import _is_device_compatible

from medusa.transforms import estimate_similarity_transform


@pytest.mark.parametrize("batch_size", [1, 100])
@pytest.mark.parametrize("n_points", [2, 3, 100])
@pytest.mark.parametrize("n_dim", [2, 3])
@pytest.mark.parametrize("estimate_scale", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_similarity_transform(batch_size, n_points, n_dim, estimate_scale, device):
    if not _is_device_compatible(device):
        return

    if n_points < n_dim:
        return

    src = np.random.randn(batch_size, n_points, n_dim).astype(np.float32)
    dst = np.random.randn(batch_size, n_points, n_dim).astype(np.float32)

    mats = np.zeros((batch_size, n_dim + 1, n_dim + 1))
    for i in range(batch_size):
        mats[i, ...] = _umeyama(src[i, ...], dst[i, ...], estimate_scale=estimate_scale)

    src = torch.from_numpy(src).to(device)
    dst = torch.from_numpy(dst).to(device)
    mats_ = estimate_similarity_transform(src, dst, estimate_scale=estimate_scale)
    mats_ = mats_.cpu().numpy()

    np.testing.assert_array_almost_equal(mats, mats_, decimal=4)


def _umeyama(src, dst, estimate_scale):
    """Copied from scikit-image (https://github.com/scikit-image/scikit-
    image/blob/main/skimage/transform/_geometric.py). to avoid having to
    install scikit-image.

    For license, please see https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt.
    """
    src = np.asarray(src)
    dst = np.asarray(dst)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T

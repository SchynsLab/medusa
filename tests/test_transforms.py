import numpy as np
import pytest
import torch
from skimage.transform._geometric import _umeyama

from medusa.transforms import estimate_similarity_transform
from conftest import _check_gha_compatible


@pytest.mark.parametrize('batch_size', [1, 100])
@pytest.mark.parametrize('n_points', [2, 3, 100])
@pytest.mark.parametrize('n_dim', [2, 3])
@pytest.mark.parametrize('estimate_scale', [True, False])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_similarity_transform(batch_size, n_points, n_dim, estimate_scale, device):
    if not _check_gha_compatible(device):
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

    if device == 'cuda':
        mats_ = mats_.cpu()

    mats_ = mats_.numpy()

    np.testing.assert_array_almost_equal(mats, mats_, decimal=4)

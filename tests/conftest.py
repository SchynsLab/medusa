import os

os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['NUMEXPR_MAX_THREADS'] = '10'

import torch
from pathlib import Path

import pytest
from medusa.defaults import LOGGER

LOGGER.setLevel('WARNING')


def _is_gha_compatible(device):
    if device == "cuda" and "GITHUB_ACTIONS" in os.environ:
        return False
    else:
        return True


def _is_pytorch3d_installed():
    try:
        import pytorch3d
        return True
    except:
        return False


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    # Code that will run after your test, for example:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

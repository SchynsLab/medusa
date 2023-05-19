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

#@pytest.fixture(autouse=True)
#def run_around_tests():
    #with torch.inference_mode():
    #    yield
    #yield
    # Code that will run after your test, for example:
    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()

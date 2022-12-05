import os


def _check_gha_compatible(device):
    if device == 'cuda' and 'GITHUB_ACTIONS' in os.environ:
        return False
    else:
        return True

import os
import pytest
from medusa.recon import videorecon
from medusa.data import get_example_video

flame_models = ['emoca-coarse', 'deca-coarse', 'deca-dense', 'emoca-dense']


@pytest.mark.parametrize("model", [*flame_models, 'mediapipe', 'fan'])
@pytest.mark.parametrize("n_frames", [None, 5])
def test_videorecon(model, n_frames):
    """ Tests the videorecon command line interface. """
    
    if model in flame_models and 'GITHUB_ACTIONS' in os.environ:
        # DECA/EMOCA models are licensed, so cannot run on GH actions
        return 

    if 'GITHUB_ACTIONS' in os.environ:
        device = 'cpu'
    else:
        device = 'cuda'

    if model in flame_models:
        try:
            from flame import DecaReconModel
        except ImportError:
            print(f"Package 'flame' is not installed; skipping test of {model}!")
            return

    vid = get_example_video(as_path=False)
    data = videorecon(vid, recon_model=model, device=device, n_frames=n_frames)

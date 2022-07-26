import os
import pytest
from pathlib import Path
from click.testing import CliRunner
from medusa.cli import videorecon_cmd
from medusa.data import get_example_video

flame_models = ['emoca-coarse', 'deca-coarse', 'deca-dense', 'emoca-dense']

@pytest.mark.parametrize("model", [*flame_models])#, 'mediapipe', 'fan'])
@pytest.mark.parametrize("n_frames", [None, 5])
def test_videorecon_cmd(model, n_frames):
    """ Tests the videorecon command line interface. """
    
    if model in flame_models and 'GITHUB_ACTIONS' in os.environ:
        # emoca is licensed, so cannot run on GH actions
        return 

    if model in flame_models:
        try:
            from flame import DecaReconModel
        except ImportError:
            print(f"Package 'flame' is not installed; skipping test of {model}!")
            return

    vid = get_example_video(as_path=False)
    runner = CliRunner()
    
    args = [vid, '-r', model, '-n', n_frames]
    
    if 'GITHUB_ACTIONS' in os.environ:
        device = 'cpu'
    else:
        device = 'cuda'

    args.extend(['--device', device])

    result = runner.invoke(videorecon_cmd, args)
    assert(result.exit_code == 0)
    expected_h5 = Path(vid.replace('.mp4', '.h5'))
    assert(expected_h5.is_file())
    expected_h5.unlink()

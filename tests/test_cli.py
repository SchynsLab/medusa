import pytest
from pathlib import Path
from click.testing import CliRunner
from medusa.cli import videorecon_cmd
from medusa.data import get_example_video


@pytest.mark.parametrize("model", ['mediapipe'])#, 'emoca'])
@pytest.mark.parametrize("n_frames", [None, 5])
@pytest.mark.parametrize("render", [True, False])
def test_videorecon_cmd(model, n_frames, render):
    """ Tests the videorecon command line interface. """
    vid = get_example_video(as_path=False)
    runner = CliRunner()
    args = [vid, '-r', model, '-n', n_frames]
    if render:
        args.extend(['--render-recon'])

    result = runner.invoke(videorecon_cmd, args)
    assert(result.exit_code == 0)
    expected_h5 = Path(vid.replace('.mp4', '_desc-recon_shape.h5'))
    assert(expected_h5.is_file())
    expected_h5.unlink()
    
    if render:
        expected_gif = Path(vid.replace('.mp4', '_desc-recon_shape.gif'))
        assert(expected_gif.is_file())
        expected_gif.unlink()
        


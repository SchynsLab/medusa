from pathlib import Path
from click.testing import CliRunner

from medusa.cli import videorecon_cmd
from medusa.data import get_example_video


def test_videorecon_cmd():
    """Tests the videorecon command line interface."""

    vid = str(get_example_video())
    runner = CliRunner()

    args = [vid, '-r', 'mediapipe', '-n', 5]

    result = runner.invoke(videorecon_cmd, args)
    assert(result.exit_code == 0)
    expected_h5 = Path(vid.replace('.mp4', '.h5'))
    assert(expected_h5.is_file())
    expected_h5.unlink()

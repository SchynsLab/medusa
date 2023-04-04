from pathlib import Path

from click.testing import CliRunner

from medusa.cli import videorecon_cmd, videorender_cmd
from medusa.data import get_example_video, get_example_data4d


def test_videorecon_cmd():
    """Tests the videorecon command line interface."""

    vid = str(get_example_video())
    runner = CliRunner()
    out = Path(__file__).parent / 'test_viz/misc/test_videorecon_cmd.h5'
    args = [vid, "-r", "mediapipe", "-n", 5, '-o', str(out)]

    result = runner.invoke(videorecon_cmd, args)
    assert result.exit_code == 0
    assert out.is_file()
    out.unlink()


def test_videorender_cli():
    """Tests the videorender command line interface."""

    data_file = str(get_example_data4d(load=False))
    runner = CliRunner()
    out = Path(__file__).parent / 'test_viz/misc/test_videorender_cmd.mp4'

    args = [data_file, '-o', str(out), '-s', 'flat']
    result = runner.invoke(videorender_cmd, args)

    assert result.exit_code == 0
    assert out.is_file()
    out.unlink()

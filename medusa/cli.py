""" Module with command-line interface functions, created using ``click``.
Each function can be accessed on the command line as ``medusa_{operation}``,
e.g. ``medusa_videorecon`` or ``medusa_align`` with arguments and options corresponding
to the function arguments, e.g.

``medusa_filter some_h5_file.h5 -l 3 -h 0.01``

To get an overview of the mandatory arguments and options, run the command with the
option ``--help``, e.g.:

```medusa_filter --help``

For more information, check out the
`documentation <https://lukas-snoek.com/medusa/api/cli.html`_.
"""

import click
from pathlib import Path

from .core import load_h5
from .preproc.recon import videorecon
from .preproc.align import align
from .preproc.resample import resample
from .preproc.filter import filter
from .preproc.epoch import epoch

# fmt: off
@click.command()
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--events-path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("-r", "--recon-model-name", default="emoca", type=click.Choice(["emoca", "mediapipe", "FAN-3D"]))
@click.option("-c", "--cfg", default=None, type=click.STRING, help="Path to recon config file")
@click.option("--device", default="cuda", type=click.Choice(["cpu", "cuda"]), help="Device to run recon on")
@click.option("-o", "--out-dir", type=click.Path(), help="Output directory")
@click.option("--render-recon", is_flag=True, help="Plot recon on video background")
@click.option("--render-on-video", is_flag=True, help="Plot recon on video background")
@click.option("--render-crop", is_flag=True, help="Render cropping results")
@click.option("-n", "--n-frames", default=None, type=click.INT, help="Number of frames to reconstruct")
def videorecon_cmd(video_path, events_path, recon_model_name, cfg, device, out_dir,
                   render_recon, render_on_video, render_crop, n_frames):
    videorecon(**locals())


@click.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option("--algorithm", default="icp", type=click.Choice(["icp", "umeyama"]))
@click.option("--qc", is_flag=True, help="Generate QC plot")
def align_cmd(data, algorithm, qc):
    align(data, algorithm, qc)


@click.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option("--sampling-freq", default=None, type=click.INT)
@click.option(
    "--kind",
    type=click.Choice(["pchip", "linear", "quadratic", "cubic"]),
    default="pchip",
)
def resample_cmd(data, sampling_freq, kind):
    resample(data, sampling_freq, kind)


@click.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option("-l", "--low-pass", default=4, help="Low-pass filter in Hz")
@click.option("-h", "--high-pass", default=0.005, help="High-pass filter in Hz")
def filter_cmd(data, low_pass, high_pass):
    filter(data, low_pass, high_pass)


@click.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option("-s", "--start", default=-0.5, help="Start of epoch")
@click.option("-e", "--end", default=3.0, help="End of epoch")
@click.option("-p", "--period", default=0.1, help="Period of epoch")
def epoch_cmd(data, start, end, period):
    epoch(data, start, end, period)


@click.command()
@click.argument("h5_path")
@click.option("-v", "--video", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--n-frames", default=None, type=click.INT, help="Number of frames to render (default is all)")
@click.option("--no-smooth", is_flag=True, help="Do not render smooth surface")
@click.option("--wireframe", is_flag=True, help="Render wireframe instead of mesh")
@click.option("--alpha", default=None, type=click.FLOAT, help="Alpha (transparency) of face")
@click.option("--scaling", default=None, type=click.FLOAT, help="Scale factor")
@click.option("--fmt", default="gif", help="Output video format")
def videorender_cmd(h5_path, video, n_frames, no_smooth, wireframe, alpha, scaling, fmt):
    """Renders the reconstructed vertices as a video."""

    h5_path = Path(h5_path)
    data = load_h5(h5_path)
    f_out = h5_path.parent / (str(h5_path.stem) + f".{fmt}")
    smooth = not no_smooth
    data.render_video(
        f_out,
        video=video,
        smooth=smooth,
        wireframe=wireframe,
        scaling=scaling,
        n_frames=n_frames,
        alpha=alpha,
    )
# fmt: on

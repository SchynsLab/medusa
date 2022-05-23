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
@click.option("--events-path", default=None, type=click.Path(exists=True, dir_okay=False),
              help='Path to events-file (a .tsv file)')
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("-r", "--recon-model-name", default="emoca", type=click.Choice(["emoca", "mediapipe", "FAN-3D"]),
              help='Name of the reconstruction model')
@click.option("-c", "--cfg", default=None, type=click.STRING,
              help="Path to a custom Medusa config file")
@click.option("--device", default="cuda", type=click.Choice(["cpu", "cuda"]),
              help="Device to run the reconstruction on (only relevant for FAN/EMOCA")
@click.option("--render-recon", is_flag=True,
              help="Render the reconstruction as a video")
@click.option("--render-on-video", is_flag=True,
              help="Render the reconstruction on top of the original video")
@click.option("--render-format", default='gif', type=click.Choice(['gif', 'mp4']),
              help="Format of rendering output")
@click.option("-n", "--n-frames", default=None, type=click.INT,
              help="Number of frames to reconstruct (5 means 'reconstruct only the 5 first frames)")
def videorecon_cmd(video_path, events_path, out, recon_model_name, cfg, device,
                   render_recon, render_on_video, render_format, n_frames):
    """ Performs frame-by-frame 3D face reconstruction of a video file."""
    data = videorecon(video_path, events_path, recon_model_name, cfg, device,
                      n_frames)

    if out is None:
        out = video_path.replace('.mp4', '')
            
    data.save(out + '.h5')

    if render_recon:
        background = video_path if render_on_video else None
        data.render_video(out + f".{render_format}", video=background)


@click.command()
@click.argument("data_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("--algorithm", default="icp", type=click.Choice(["icp", "umeyama"]),
              help="Name of the alignment algorithm")
@click.option("--additive-alignment", is_flag=True,
              help="Whether to perform alignment on top of existing transform")
@click.option("--ignore-existing", is_flag=True,
              help="Whether to ignore existing alignment and run alignment")
@click.option("--qc", is_flag=True,
              help="Generate a QC figure with motion and PCA traces")
@click.option("--reference-index", default=0, 
              help="Index of reference mesh to align other meshes to (default = 0 = first")
def align_cmd(data_file, out, algorithm, additive_alignment, ignore_existing, qc,
              reference_index):
    """ Performs alignment ("motion correction") of a mesh time series. """
    data = align(data_file, algorithm, additive_alignment, ignore_existing,
                 reference_index)

    if out is None:
        out = data_file.replace('.h5', '')

    data.save(out + '.h5')

    if qc:
        data.plot_data(out + '.png', plot_motion=True, plot_pca=True, n_pca=3)


@click.command()
@click.argument("data_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("--sampling-freq", default=None, type=click.INT,
              help="Desired sampling frequency of the data (an integer, in hertz)")
@click.option("--kind", default="pchip", type=click.Choice(["pchip", "linear", "quadratic", "cubic"]),
              help="Kind of interpolation used")
def resample_cmd(data_file, out, sampling_freq, kind):
    """ Performs temporal resampling of a mesh time series. """
    
    data = resample(data_file, sampling_freq, kind)

    if out is None:
        out = data_file.replace('.h5', '')

    data.save(out + '.h5')


@click.command()
@click.argument("data_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("-l", "--low-pass", default=4,
              help="Low-pass filter in hertz")
@click.option("-h", "--high-pass", default=0.005,
              help="High-pass filter in hertz")
def filter_cmd(data_file, low_pass, high_pass):
    """ Performs temporal filtering of a mesh time series. """

    data = filter(data_file, low_pass, high_pass)

    if out is None:
        out = data_file.replace('.h5', '')

    data.save(out + '.h5')


@click.command()
@click.argument("data_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("-s", "--start", default=-0.5,
              help="Start of epoch (in seconds), relative to event onset")
@click.option("-e", "--end", default=3.0,
              help="End of epoch (in seconds), relative to event onset")
@click.option("-p", "--period", default=0.1,
              help="Desired period (1 / sampling frequency) of epoch (in seconds)")
def epoch_cmd(data_file, start, end, period):
    """ Performs epoching of a mesh time series. """

    data = epoch(data_file, start, end, period)

    if out is None:
        out = data_file.replace('.h5', '')

    data.save(out + '.h5')


@click.command()
@click.argument("data_file")
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("-v", "--video", type=click.Path(exists=True, dir_okay=False),
              help="Path to video file, when rendering on top of original video")
@click.option("-n", "--n-frames", default=None, type=click.INT,
              help="Number of frames to render (default is all)")
@click.option("--no-smooth", is_flag=True,
              help="Do not render smooth surface")
@click.option("--wireframe", is_flag=True,
              help="Render wireframe instead of mesh")
@click.option("--alpha", default=None, type=click.FLOAT,
              help="Alpha (transparency) of face")
@click.option("--scaling", default=None, type=click.FLOAT,
              help="Scale factor of rendered video (e.g., 0.25 = 25% of original size")
@click.option("--render-format", default="gif",
              help="Output format of rendered video (gif or mp4)")
def videorender_cmd(data_file, out, video, n_frames, no_smooth, wireframe, alpha, scaling, render_format):
    """ Renders the reconstructed mesh time series as a video (gif or mp4)."""

    data = load_h5(data_file)
    h5_path = Path(h5_path)
    
    if out is None:
        out = data_file.replace('.h5', '')

    smooth = not no_smooth
    data.render_video(
        out + f'.{render_format}',
        video=video,
        smooth=smooth,
        wireframe=wireframe,
        scaling=scaling,
        n_frames=n_frames,
        alpha=alpha,
    )
# fmt: on

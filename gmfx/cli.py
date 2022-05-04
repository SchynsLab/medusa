# Module with command-line interface functions, created
# using `click`. Each function can be accessed on the command
# line as `gmfx_{operation}`, e.g. `gmfx_recon` or
# `gmfx_align` with arguments and options corresponding
# to the function arguments, e.g.
#
# `gmfx_filter some_h5_file.h5 -l 3 -h 0.01`

import click
import shutil
from pathlib import Path
from .preproc.recon import videorecon
from .preproc.align import align
from .preproc.resample import resample
from .preproc.filter import filter


@click.command()
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-e', '--events-path', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('-r', '--recon-model-name', default='emoca', type=click.Choice(['emoca', 'mediapipe', 'FAN-3D']))
@click.option('-c', '--cfg', default=None, type=click.STRING, help='Path to recon config file')
@click.option('--device', default='cuda', type=click.Choice(['cpu', 'cuda']), help='Device to run recon on')
@click.option('-o', '--out-dir', type=click.Path(), help='Output directory')
@click.option('--render-on-video', is_flag=True, help='Plot recon on video background')
def videorecon_cmd(video_path, events_path, recon_model_name, cfg, device, out_dir, render_on_video):
    videorecon(**locals())


@click.command()
@click.argument('data', type=click.Path(exists=True, dir_okay=False))
@click.option('--algorithm', default='icp', type=click.Choice(['icp', 'umeyama']))
@click.option('--video', default=None, type=click.Path(exists=True, dir_okay=False), help='Video for rendering')
def align_cmd(data, algorithm, video):
    align(**locals())


@click.command()
@click.argument('data', type=click.Path(exists=True, dir_okay=False))
@click.option('--sampling-freq', default=None, type=click.INT)
@click.option('--kind', type=click.Choice(['pchip', 'linear', 'quadratic', 'cubic']), default='pchip')
def resample_cmd(data, sampling_freq, kind):
    resample(**locals())


@click.command()
@click.argument('data', type=click.Path(exists=True, dir_okay=False))
@click.option('-l', '--low-pass', default=4, help="Low-pass filter in Hz")
@click.option('-h', '--high-pass', default=0.005, help="High-pass filter in Hz")
def filter_cmd(data, low_pass, high_pass):
    filter(**locals())


@click.command()
def create_cfg_cmd():
    src = Path(__file__).parent / 'configs/deca.yaml'
    dst = Path('.') / 'deca.yaml'
    shutil.copyfile(src, dst)
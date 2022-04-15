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
from .preproc.recon import recon
from .preproc.align import align
from .preproc.resample import resample
from .preproc.filter import filter


@click.command()
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-e', '--events-path', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('-c', '--cfg', default=None, type=click.STRING, help='Path to recon config file')
@click.option('--device', default='cuda', type=click.Choice(['cpu', 'cuda']), help='Device to run recon on')
@click.option('-o', '--out-dir', type=click.Path(), help='Output directory')
@click.option('--allow-pose', is_flag=True, help='Allow pose to influence mesh')
def recon_cmd(video_path, events_path, cfg, device, out_dir, allow_pose):
    recon(**locals())


@click.command()
@click.argument('data', type=click.Path(exists=True, dir_okay=False))
@click.option('--algorithm', default='icp', type=click.Choice(['icp', 'umeyama']))
@click.option('--device', default='cuda', type=click.Choice(['cpu', 'cuda']), help='Device to run recon on')
def align_cmd(data, algorithm, device):
    align(**locals())


@click.command()
@click.argument('data', type=click.Path(exists=True, dir_okay=False))
@click.option('--sampling-rate', default=None, type=click.INT)
@click.option('--kind', type=click.Choice(['pchip', 'linear', 'quadratic', 'cubic']), default='pchip')
@click.option('--device', default='cuda', type=click.Choice(['cpu', 'cuda']), help='Device to run recon on')
def resample_cmd(data, sampling_rate, kind, device):
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
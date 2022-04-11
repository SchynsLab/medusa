# Module with command-line interface functions, created
# using `click`. Each function can be accessed on the command
# line as `gmfx_{operation}`, e.g. `gmfx_recon` or
# `gmfx_align` with arguments and options corresponding
# to the function arguments, e.g.
#
# `gmfx_filter output_directory -p sub-01 -l 3 -h 0.01`

import click
import shutil
from pathlib import Path
from .preproc.recon import recon
from .preproc.align import align
from .preproc.interpolate import interpolate
from .preproc.filter import filter


@click.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--out-dir', default=None, type=click.Path(), help='Output directory')
@click.option('-p', '--participant-label', default='sub-01', help='Participant identifier (e.g., sub-01)')
@click.option('-c', '--cfg', default=None, type=click.STRING, help='Path to recon config file')
@click.option('--device', default='cuda', type=click.Choice(['cpu', 'cuda']), help='Device to run recon on')
def recon_cmd(in_dir, out_dir, participant_label, cfg, device):
    recon(**locals())


@click.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.option('-p', '--participant-label', default='sub-01', help='Participant identifier (e.g., sub-01)')
@click.option('--ref-verts', default='eyes+nose', type=click.Choice(['all', 'eyes+nose', 'scalp']), help='Which vertices to use for alignment')
@click.option('--save-all', is_flag=True, help='Save all intermediate output for debugging')
def align_cmd(in_dir, participant_label, ref_verts, save_all):
    align(**locals())


@click.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.option('-p', '--participant-label', default='sub-01', help='Participant identifier (e.g., sub-01)')
@click.option('--save-all', is_flag=True, help='Save all intermediate output for debuggin')
def interpolate_cmd(in_dir, participant_label, save_all):
    interpolate(**locals())


@click.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.option('-p', '--participant-label', default='sub-01', help='Participant identifier (e.g., sub-01)')
@click.option('-l', '--low-pass', default=2.5, help="Lowpass filter in Hz")
@click.option('-h', '--high-pass', default=0.01, help="Highpass filter in Hz")
@click.option('--save-all', is_flag=True, help='Save all intermediate output for debuggin')
def filter_cmd(in_dir, participant_label, low_pass, high_pass, save_all):
    filter(**locals())


@click.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_dir', type=click.Path())
@click.option('-p', '--participant-label', default='sub-01', help='Participant identifier (e.g., sub-01)')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda']), help='Device to run recon on')
def preproc_cmd(in_dir, out_dir, participant_label, device):
    raise NotImplementedError
    #recon(in_dir, out_dir, participant_label, device)
    #align()
    #interpolate()
    #filter()


@click.command()
def create_cfg_cmd():
    src = Path(__file__).parent / 'configs/deca.yaml'
    dst = Path('.') / 'deca.yaml'
    shutil.copyfile(src, dst)
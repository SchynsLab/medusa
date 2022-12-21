"""Module with command-line interface functions, created using ``click``. Each
function can be accessed on the command line as ``medusa_{operation}``, e.g.
``medusa_videorecon`` or ``medusa_align`` with arguments and options
corresponding to the function arguments, e.g.

``medusa_filter some_h5_file.h5 -l 3 -h 0.01``

To get an overview of the mandatory arguments and options, run the command with the
option ``--help``, e.g.:

```medusa_filter --help``

For more information, check out the
`documentation <https://lukas-snoek.com/medusa/api/cli.html`_.
"""

import logging
import shutil
import zipfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import click
import torch
import yaml
import gdown

from .constants import DEVICE
from .io import download_file
from .log import get_logger
from .containers import Data4D
from .preproc.align import align
from .preproc.epoch import epoch
from .preproc.filter import bw_filter
from .preproc.resample import resample
from .recon import videorecon

RECON_MODELS = [
    "spectre-coarse",
    "emoca-dense",
    "emoca-coarse",
    "deca-dense",
    "deca-coarse",
    "mediapipe",
]

# fmt: off
@click.command()
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("-r", "--recon-model", default="emoca-coarse", type=click.Choice(RECON_MODELS),
              help='Name of the reconstruction model')
@click.option("--device", default="cuda", type=click.Choice(["cpu", "cuda"]),
              help="Device to run the reconstruction on (only relevant for EMOCA")
@click.option('--n-frames', '-n', default=None, type=click.INT,
              help='Number of frames to process')
@click.option('--batch-size', '-b', default=32, type=click.INT,
              help='Batch size of inputs to recon model')
def videorecon_cmd(video_path, out, recon_model, device, n_frames, batch_size):
    """Performs frame-by-frame 3D face reconstruction of a video file."""

    data = videorecon(video_path=video_path, recon_model=recon_model, device=device,
                     n_frames=n_frames, batch_size=batch_size)

    if out is None:
       ext = Path(video_path).suffix
       out = video_path.replace(ext, '.h5')

    out = Path(out)
    if out.suffix != '.h5':
       raise ValueError("Extension of output file should be '.h5'!")

    data.save(out)


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
@click.option("--reference-index", default=0,
              help="Index of reference mesh to align other meshes to (default = 0 = first")
def align_cmd(data_file, out, algorithm, additive_alignment, ignore_existing,
              reference_index):
    """Performs alignment ("motion correction") of a mesh time series."""
    data = align(data_file, algorithm, additive_alignment, ignore_existing,
                 reference_index)

    if out is None:
        out = data_file.replace('.h5', '')

    data.save(out + '.h5')


@click.command()
@click.argument("data_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("--sampling-freq", default=None, type=click.INT,
              help="Desired sampling frequency of the data (an integer, in hertz)")
@click.option("--kind", default="pchip", type=click.Choice(["pchip", "linear", "quadratic", "cubic"]),
              help="Kind of interpolation used")
def resample_cmd(data_file, out, sampling_freq, kind):
    """Performs temporal resampling of a mesh time series."""

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
def filter_cmd(data_file, out, low_pass, high_pass):
    """Performs temporal filtering of a mesh time series."""

    data = bw_filter(data_file, low_pass, high_pass)

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
@click.option("--baseline-correct", is_flag=True,
              help="Whether to perform baseline correction")
@click.option("--add-back-grand-mean", is_flag=True,
              help="Whether to add back the grand mean after baseline correction")
@click.option("--to-mne", is_flag=True,
              help="Whether convert the output to an MNE EpochsArray object")
def epoch_cmd(data_file, out, start, end, period, baseline_correct, add_back_grand_mean,
              to_mne):
    """Performs epoching of a mesh time series."""

    data = Data4D.load(data_file)
    epochsarray = epoch(data, start, end, period, baseline_correct,
                        add_back_grand_mean=add_back_grand_mean)

    if out is None:
        out = data_file.replace('.h5', '')

    if to_mne:
        epochsarray = epochsarray.to_mne(frame_t=data.frame_t, include_global_motion=True)
        epochsarray.save(out + '_epo.fif', split_size="2GB", fmt="single",
                         overwrite=True, split_naming="bids", verbose="WARNING")
    else:
        epochsarray.save(out + '_epo.h5')


@click.command()
@click.argument("data_file")
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("-v", "--video", type=click.Path(exists=True, dir_okay=False),
              help="Path to video file, when rendering on top of original video")
@click.option("-n", "--n-frames", default=None, type=click.INT,
              help="Number of frames to render (default is all)")
@click.option("--smooth", is_flag=True,
              help="Render smooth surface")
@click.option("--wireframe", is_flag=True,
              help="Render wireframe instead of mesh")
@click.option("--alpha", default=None, type=click.FLOAT,
              help="Alpha (transparency) of face")
@click.option("--scale", default=None, type=click.FLOAT,
              help="Scale factor of rendered video (e.g., 0.25 = 25% of original size")
def videorender_cmd(data_file, out, video, n_frames, smooth, wireframe, alpha, scale):
    """Renders the reconstructed mesh time series as a video (gif or mp4)."""

    data = load_h5(data_file)

    if out is None:
        out = data_file.replace('.h5', '.mp4')

    out = Path(out)
    data.render_video(
        out,
        video=video,
        smooth=smooth,
        wireframe=wireframe,
        scale=scale,
        n_frames=n_frames,
        alpha=alpha,
    )
# fmt: on


@click.command()
@click.option("--directory", default="./medusa_ext_data")
@click.option("--overwrite", is_flag=True)
@click.option(
    "--username", default=None, type=click.STRING, help="Username for FLAME website"
)
@click.option(
    "--password", default=None, type=click.STRING, help="Password for FLAME website"
)
@click.option("--device", default=DEVICE, type=click.STRING)
@click.option("--no-validation", is_flag=True)
def download_ext_data(directory, overwrite, username, password, device, no_validation):
    """Command-line utility to download external data."""

    click.secho(
        """
    This command will download the external data and models necessary to run some of the models in Medusa,
    including all models based on the FLAME topology. To use this data and models, you need to register
    at the websites where the data/models are hosted and agree to their license terms *prior* to running
    this command. This command will prompt you to confirm that you've registered for each model separately.
    In addition, to download the FLAME model itself (necessary for any FLAME-based reconstruction model),
    you need to pass your username (--username arg) and password (--passwd arg) of your account on their
    website (https://flame.is.tue.mpg.de) to this command, for example:

    medusa_download_ext_data --username test@gmail.com --password yourpassword
    """,
        fg="red",
        bold=True,
        blink=True,
    )

    logger = get_logger("INFO")
    logger.info(f"Downloading models to {directory}, configuring to run on {device}")

    directory = Path(directory)
    if not directory.is_dir():
        logger.info(f"Creating output directory {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    if username is not None and password is not None:
        logger.info("FLAME: starting download ...")
        url = "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1"
        data = {"username": username, "password": password}
        f_out = directory / "FLAME2020.zip"
        download_file(url, f_out, data=data, verify=True, overwrite=overwrite)
        url = "https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip"
        f_out = directory / "FLAME_masks.zip"
        download_file(url, f_out, overwrite=overwrite, cmd_type="get")
    else:
        logger.warning("FLAME: cannot download, because no username and/or password!")

    desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")
    if click.confirm(
        f"{desc} DECA: I have registered and agreed to the license terms at https://deca.is.tue.mpg.de"
    ):
        # url = 'https://download.is.tue.mpg.de/download.php?domain=deca&resume=1&sfile=deca_model.tar'
        f_out = directory / "deca_model.tar"
        if not f_out.is_file() or overwrite:
            # download_file(url, f_out, overwrite=overwrite, cmd_type='get')
            url = "https://drive.google.com/u/0/uc?id=1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje"
            gdown.download(url, str(f_out), quiet=True)

    desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")
    if click.confirm(
        f"{desc} EMOCA: I have registered and agreed to the license terms at https://emoca.is.tue.mpg.de"
    ):
        url = "https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA.zip"
        f_out = directory / "EMOCA.zip"
        download_file(url, f_out, overwrite=overwrite)

        # Note to self: not sure whether the EMOCA.zip data contains the finetuned detail
        # encoder or the DECA.zip file (below) ...
        # url = "https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/DECA.zip"
        # f_out = directory / 'DECA_for_EMOCA.zip'
        # download_file(url, f_out, overwrite=overwrite)

    desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ] ")
    if click.confirm(
        f"{desc} MICA: I agree to the license terms at https://github.com/Zielon/MICA/blob/master/LICENSE"
    ):
        url = "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1"
        f_out = directory / "mica.tar"
        download_file(url, f_out, overwrite=overwrite, cmd_type="get")

    f_out = directory / "spectre_model.tar"
    if not f_out.is_file() or overwrite:
        logger.info("SPECTRE: starting download ...")
        url = "https://drive.google.com/u/0/uc?id=1vmWX6QmXGPnXTXWFgj67oHzOoOmxBh6B&export=download"
        gdown.download(url, str(f_out), quiet=True)

    if no_validation:
        return

    cfg = {}  # to be saved later

    data_dir = Path(directory).resolve()
    if not data_dir.is_dir():
        logger.exception(f"Directory '{directory}' does not exist!")
        exit()

    flame_model_path = data_dir / "FLAME/generic_model.pkl"
    if flame_model_path.is_file():
        logger.info("FLAME model is ready to use!")
        cfg["flame_path"] = str(flame_model_path)
    else:
        flame_zip = data_dir / "FLAME2020.zip"
        if not flame_zip.is_file():
            logger.warning(f"File '{str(flame_zip)}' does not exist!")
        else:
            with zipfile.ZipFile(flame_zip, "r") as zip_ref:
                zip_ref.extractall(directory / "FLAME/")

            logger.info("FLAME model is ready to use!")
            cfg["flame_path"] = str(flame_model_path)
            flame_zip.unlink()

    flame_masks_path = data_dir / "FLAME/FLAME_masks.pkl"
    if flame_masks_path.is_file():
        logger.info("FLAME masks are ready to use!")
        cfg["flame_masks_path"] = str(flame_masks_path)
    else:
        flame_masks_zip = data_dir / "FLAME_masks.zip"
        if not flame_masks_zip.is_file():
            logger.warning(f"File '{str(flame_masks_zip)}' does not exist!")
        else:
            with zipfile.ZipFile(flame_masks_zip, "r") as zip_ref:
                zip_ref.extractall(directory / "FLAME/")

            logger.info("FLAME masks are ready to use!")
            flame_masks_zip.unlink()
            cfg["flame_masks_path"] = str(flame_masks_path)

    deca_model_path = data_dir / "deca_model.tar"
    if deca_model_path.is_file():
        logger.info("DECA model is ready to use!")
        cfg["deca_path"] = str(deca_model_path)
    else:
        logger.warning(f"File {deca_model_path} does not exist!")

    ckpt_out = data_dir / "emoca.ckpt"
    if ckpt_out.is_file():
        logger.info("EMOCA model is ready to use!")
        cfg["emoca_path"] = str(ckpt_out)
    else:
        logger.info(f"Configuring EMOCA for device '{device}'!")
        emoca_zip = data_dir / "EMOCA.zip"
        if not emoca_zip.is_file():
            logger.warning(f"File '{str(emoca_zip)}' does not exist!")
        else:
            logger.info("Unzipping EMOCA.zip file ...")
            with zipfile.ZipFile(emoca_zip, "r") as zip_ref:
                zip_ref.extractall(f"{directory}/")

            cfg_ = data_dir / "EMOCA/cfg.yaml"
            if cfg_.is_file():
                cfg_.unlink()

            ckpt = list(data_dir.glob("**/*.ckpt"))
            if len(ckpt) == 0:
                logger.exception("Could not find EMOCA .ckpt file!", exc_info=False)
                exit()

            ckpt = ckpt[0]
            shutil.move(ckpt, data_dir / "emoca.ckpt")
            shutil.rmtree(data_dir / "EMOCA")

            logger.info("Reorganizing EMOCA checkpoint file ... ")
            sd = torch.load(data_dir / "emoca.ckpt", map_location=device)["state_dict"]
            models = ["E_flame", "E_detail", "E_expression", "D_detail"]

            state_dict = {}
            for mod in models:
                state_dict[mod] = OrderedDict()

                for key, value in sd.items():
                    if mod in key:
                        k = key.split(mod + ".")[1]
                        state_dict[mod][k] = value.to(device=device)

            torch.save(state_dict, ckpt_out)
            logger.info(f"EMOCA model is ready to use!")
            cfg["emoca_path"] = str(ckpt_out)

    mica_model_path = data_dir / "mica.tar"
    if mica_model_path.is_file():
        logger.info("MICA model is ready to use!")
        cfg["mica_path"] = str(mica_model_path)
    else:
        logger.warning(f"File {mica_model_path} does not exist!")

    spectre_model_path = data_dir / "spectre_model.tar"
    if spectre_model_path.is_file():
        logging.info("Spectre model is ready to use!")
        cfg["spectre_path"] = str(spectre_model_path)
    else:
        logger.warning(f"File {spectre_model_path} does not exist!")

    cfg_path = Path(__file__).parent / "data/flame/config.yaml"
    with open(cfg_path, "w") as f_out:
        logger.info(f"Saving config file to {cfg_path}!")
        yaml.dump(cfg, f_out, default_flow_style=False)

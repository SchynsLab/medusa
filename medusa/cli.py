"""Module with command-line interface functions, created using ``click``. Each
function can be accessed on the command line as ``medusa_{operation}``, e.g.
``medusa_videorecon``.

To get an overview of the mandatory arguments and options, run the command with the
option ``--help``, e.g.:

```medusa_videorecon --help``

For more information, check out the
`documentation <https://lukas-snoek.com/medusa/api/cli.html`_.
"""

import shutil
import zipfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import click
import gdown
import torch
import yaml

from .defaults import DEVICE, LOGGER, RECON_MODELS
from .containers import Data4D
from .io import download_file
from .recon import videorecon


@click.command()
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default=None, type=click.Path(),
              help="File to save output to (shouldn't have an extension)")
@click.option("-r", "--recon-model", default="emoca-coarse", type=click.Choice(RECON_MODELS),
              help='Name of the reconstruction model')
@click.option("--device", default=DEVICE, type=click.Choice(["cpu", "cuda"]),
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
@click.argument("data_file")
@click.option("-o", "--out", default=None, type=click.Path(), help="File to save output to (shouldn't have an extension)")
@click.option("-v", "--video", type=click.Path(exists=True, dir_okay=False), help="Path to video file, when rendering on top of original video")
@click.option('-r', '--renderer', default='pyrender', type=click.Choice(['pyrender', 'pytorch3d']))
@click.option("-s", "--shading", default='flat', type=click.Choice(['flat', 'smooth']), help="Type of shading")
@click.option("--alpha", default=None, type=click.FLOAT, help="Alpha (transparency) of face")
@click.option("--device", default=DEVICE, type=click.Choice(["cpu", "cuda"]),
              help="Device to run the rendering on")
def videorender_cmd(data_file, out, video, renderer, shading, alpha, device):
    """Renders the reconstructed mesh time series as a video (gif or mp4)."""

    data = Data4D.load(data_file, device=device)

    if out is None:
        out = data_file.replace('.h5', '.mp4')

    data.render_video(
        Path(out), video=video, renderer=renderer, shading=shading, alpha=alpha,
    )


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

    LOGGER.info(f"Downloading models to {directory}, configuring to run on {device}")

    directory = Path(directory)
    if not directory.is_dir():
        LOGGER.info(f"Creating output directory {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    if username is not None and password is not None:
        LOGGER.info("FLAME: starting download ...")
        url = "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1"
        data = {"username": username, "password": password}
        f_out = directory / "FLAME2020.zip"
        download_file(url, f_out, data=data, verify=True, overwrite=overwrite)
        url = "https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip"
        f_out = directory / "FLAME_masks.zip"
        download_file(url, f_out, overwrite=overwrite, cmd_type="get")
    else:
        LOGGER.warning("FLAME: cannot download, because no username and/or password!")

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
        LOGGER.info("SPECTRE: starting download ...")
        url = "https://drive.google.com/u/0/uc?id=1vmWX6QmXGPnXTXWFgj67oHzOoOmxBh6B&export=download"
        gdown.download(url, str(f_out), quiet=True)

    f_out = directory / 'buffalo_l/det_10g.onnx'
    if not f_out.is_file() or overwrite:
        LOGGER.info("INSIGHTFACE: starting download 'buffalo_l.zip' ...")
        f_out = f_out.parents[1] / "buffalo_l.zip"
        url = "https://drive.google.com/u/0/uc?id=1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB&export=download"
        gdown.download(url, str(f_out), quiet=True)
        gdown.extractall(str(f_out))
        f_out.unlink()

    ### STARTING VALIDATION ###
    if no_validation:
        return

    cfg = {}  # to be saved later

    data_dir = Path(directory).resolve()
    if not data_dir.is_dir():
        LOGGER.exception(f"Directory '{directory}' does not exist!")
        exit()

    flame_model_path = data_dir / "FLAME/generic_model.pkl"
    if flame_model_path.is_file():
        LOGGER.info("FLAME model is ready to use!")
        cfg["flame_path"] = str(flame_model_path)
    else:
        flame_zip = data_dir / "FLAME2020.zip"
        if not flame_zip.is_file():
            LOGGER.warning(f"File '{str(flame_zip)}' does not exist!")
        else:
            with zipfile.ZipFile(flame_zip, "r") as zip_ref:
                zip_ref.extractall(directory / "FLAME/")

            LOGGER.info("FLAME model is ready to use!")
            cfg["flame_path"] = str(flame_model_path)
            flame_zip.unlink()

    flame_masks_path = data_dir / "FLAME/FLAME_masks.pkl"
    if flame_masks_path.is_file():
        LOGGER.info("FLAME masks are ready to use!")
        cfg["flame_masks_path"] = str(flame_masks_path)
    else:
        flame_masks_zip = data_dir / "FLAME_masks.zip"
        if not flame_masks_zip.is_file():
            LOGGER.warning(f"File '{str(flame_masks_zip)}' does not exist!")
        else:
            with zipfile.ZipFile(flame_masks_zip, "r") as zip_ref:
                zip_ref.extractall(directory / "FLAME/")

            LOGGER.info("FLAME masks are ready to use!")
            flame_masks_zip.unlink()
            cfg["flame_masks_path"] = str(flame_masks_path)

    deca_model_path = data_dir / "deca_model.tar"
    if deca_model_path.is_file():
        LOGGER.info("DECA model is ready to use!")
        cfg["deca_path"] = str(deca_model_path)
    else:
        LOGGER.warning(f"File {deca_model_path} does not exist!")

    ckpt_out = data_dir / "emoca.ckpt"
    if ckpt_out.is_file():
        LOGGER.info("EMOCA model is ready to use!")
        cfg["emoca_path"] = str(ckpt_out)
    else:
        LOGGER.info(f"Configuring EMOCA for device '{device}'!")
        emoca_zip = data_dir / "EMOCA.zip"
        if not emoca_zip.is_file():
            LOGGER.warning(f"File '{str(emoca_zip)}' does not exist!")
        else:
            LOGGER.info("Unzipping EMOCA.zip file ...")
            with zipfile.ZipFile(emoca_zip, "r") as zip_ref:
                zip_ref.extractall(f"{directory}/")

            cfg_ = data_dir / "EMOCA/cfg.yaml"
            if cfg_.is_file():
                cfg_.unlink()

            ckpt = list(data_dir.glob("**/*.ckpt"))
            if len(ckpt) == 0:
                LOGGER.exception("Could not find EMOCA .ckpt file!", exc_info=False)
                exit()

            ckpt = ckpt[0]
            shutil.move(ckpt, data_dir / "emoca.ckpt")
            shutil.rmtree(data_dir / "EMOCA")

            LOGGER.info("Reorganizing EMOCA checkpoint file ... ")
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
            LOGGER.info(f"EMOCA model is ready to use!")
            cfg["emoca_path"] = str(ckpt_out)

    mica_model_path = data_dir / "mica.tar"
    if mica_model_path.is_file():
        LOGGER.info("MICA model is ready to use!")
        cfg["mica_path"] = str(mica_model_path)
    else:
        LOGGER.warning(f"File {mica_model_path} does not exist!")

    spectre_model_path = data_dir / "spectre_model.tar"
    if spectre_model_path.is_file():
        LOGGER.info("Spectre model is ready to use!")
        cfg["spectre_path"] = str(spectre_model_path)
    else:
        LOGGER.warning(f"File {spectre_model_path} does not exist!")

    buffalo_path = data_dir / 'buffalo_l'
    if buffalo_path.is_dir():
        LOGGER.info("Insightface models are ready to use!")
        cfg["buffalo_path"] = str(buffalo_path)
    else:
        LOGGER.warning(f"Insightface buffalo data do not exist!")

    cfg_path = Path(__file__).parent / "data/config.yaml"
    with open(cfg_path, "w") as f_out:
        LOGGER.info(f"Saving config file to {cfg_path}!")
        yaml.dump(cfg, f_out, default_flow_style=False)

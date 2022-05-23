import os
import click
import torch
import shutil
import zipfile
from pathlib import Path
from collections import OrderedDict
from medusa.utils import get_logger


@click.command()
@click.option('--cpu', is_flag=True, help='Model will be run on CPU only')
def main(cpu):

    logger = get_logger()

    data_dir = Path('./ext_data')
    if not data_dir.is_dir():
        logger.exception("Directory './ext_data' does not exist!")
        exit()

    flame_model_path = data_dir / 'FLAME/generic_model.pkl'
    if flame_model_path.is_file():
        logger.info("FLAME directory already configured!")
    else:
        flame_zip = data_dir / 'FLAME2020.zip'
        if not flame_zip.is_file():
            logger.exception("File './ext_data/FLAME2020.zip' does not exist!")
            exit()

        logger.info("Unzipping FLAME2020.zip file ...")
        with zipfile.ZipFile(flame_zip, 'r') as zip_ref:
            zip_ref.extractall('./ext_data/FLAME/')

    logger.info("FLAME is downloaded and configured correctly!")

    device = 'cpu' if cpu else 'cuda'
    ckpt_out = data_dir / 'EMOCA/emoca.ckpt'
    if ckpt_out.is_file():
        logger.info("EMOCA seems to be already configured!", exc_info=False)
    else:
        logger.info(f"Configuring EMOCA for device '{device}'!")
        emoca_zip = data_dir / 'EMOCA.zip'
        if not emoca_zip.is_file():
            logger.exception("File './ext_data/EMOCA.zip' does not exist!")
            exit()

        logger.info("Unzipping EMOCA.zip file ...")
        with zipfile.ZipFile(emoca_zip, 'r') as zip_ref:
            zip_ref.extractall('./ext_data/')

        cfg = data_dir / 'EMOCA/cfg.yaml'
        if cfg.is_file():
            cfg.unlink()

        ckpt = list(data_dir.glob("**/*.ckpt"))
        if len(ckpt) == 0:
            logger.exception("Could not find EMOCA .ckpt file!", exc_info=False)
            exit()

        ckpt = ckpt[0]
        shutil.move(ckpt, data_dir / "EMOCA/emoca.ckpt")
        detail_dir = data_dir / "EMOCA/detail"
        if detail_dir.is_dir():
            shutil.rmtree(detail_dir)

        logger.info("Reorganizing EMOCA checkpoint file ... ")
        sd = torch.load(data_dir / "EMOCA/emoca.ckpt")["state_dict"]
        models = ["E_flame", "E_detail", "E_expression", "D_detail"]

        state_dict = {}
        for mod in models:
            state_dict[mod] = OrderedDict()

            for key, value in sd.items():
                if mod in key:
                    k = key.split(mod + ".")[1]
                    state_dict[mod][k] = value.to(device=device)

        torch.save(state_dict, ckpt_out)
        logger.info(f"Saving reorganized checkpoint file at {str(ckpt_out)}")
    
    logger.info("EMOCA is downloaded and configured correctly!")


if __name__ == '__main__':
    main()
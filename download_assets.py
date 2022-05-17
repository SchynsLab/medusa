import os, shutil, torch
import requests, zipfile, io
from pathlib import Path
from collections import OrderedDict

data_dir = Path(__file__).parent / "ext_data"
data_dir.mkdir(exist_ok=True)

if not (data_dir / "EMOCA/emoca.ckpt").exists():
    print("Downloading EMOCA ... ", end="")
    r = requests.get(
        "https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA.zip"
    )
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(str(data_dir))
    print("done.")

    os.remove(data_dir / "EMOCA/cfg.yaml")
    ckpt = list(data_dir.glob("**/*.ckpt"))[0]
    shutil.move(ckpt, data_dir / "EMOCA/emoca.ckpt")
    shutil.rmtree(data_dir / "EMOCA/detail")

    print("Reorganizing EMOCA checkpoint ... ", end="")
    sd = torch.load(data_dir / "EMOCA/emoca.ckpt")["state_dict"]
    models = ["E_flame", "E_detail", "E_expression", "D_detail"]

    state_dict = {}
    for mod in models:
        state_dict[mod] = OrderedDict()

        for key, value in sd.items():
            if mod in key:
                k = key.split(mod + ".")[1]
                state_dict[mod][k] = value.to(device="cuda")

    torch.save(state_dict, data_dir / "EMOCA/emoca.ckpt")
    print("done.")
else:
    print("EMOCA already downloaded!")

if not (data_dir / "FLAME").exists():
    print("Downloading FLAME ... ", end="")
    r = requests.get("https://download.is.tue.mpg.de/emoca/assets/FLAME.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(str(data_dir))
    print("done.")
else:
    print("FLAME already downloaded!")

print(
    "\nTo download the BFM texture mesh, check out:\n"
    "https://github.com/TimoBolkart/BFM_to_FLAME\n"
    "Place the file (FLAME_albedo_from_BFM.npz) in ext_data/FLAME/"
)

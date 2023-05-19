from pathlib import Path
from timer import FancyTimer

from medusa.recon import DecaReconModel
from medusa.data import get_example_image

timer_ = FancyTimer()
params = {
    "model_name": ['deca-coarse'],#, 'emoca-coarse'],
    "device": ["cuda"],
    "batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
}

for p in timer_.iter(params):
    model = DecaReconModel(name=p["model_name"], device=p["device"])
    img = get_example_image(load_torch=True, device=p["device"])
    img = img.repeat(p["batch_size"], 1, 1, 1)
    timer_.time(model, [img], n_warmup=1, repeats=5, params=p)

df = timer_.to_df()
print(df.groupby(list(params.keys())).mean())
f_out = Path(__file__).parent / "speed_benchmark_recon_flame.tsv"
df.to_csv(f_out, sep="\t", index=False)

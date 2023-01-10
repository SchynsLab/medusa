from pathlib import Path

from timer import FancyTimer

from medusa.crop import AlignCropModel, BboxCropModel
from medusa.data import get_example_frame

timer_ = FancyTimer()
params = {
    "model_cls": [BboxCropModel, AlignCropModel],
    "device": ["cpu", "cuda"],
    "batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
}

for p in timer_.iter(params):
    model = p["model_cls"](device=p["device"])
    img = get_example_frame(load_torch=True, device=p["device"])
    img = img.repeat(p["batch_size"], 1, 1, 1)
    timer_.time(model, [img], n_warmup=2, repeats=20, params=p)

df = timer_.to_df()
print(df.groupby(list(params.keys())).mean())
f_out = Path(__file__).parent / "speed_benchmark_crop.tsv"
df.to_csv(f_out, sep="\t", index=False)

from pathlib import Path

from timer import FancyTimer

from medusa.data import get_example_image
from medusa.detect import SCRFDetector, YunetDetector

timer_ = FancyTimer()
params = {
    "model_cls": [SCRFDetector],#, YunetDetector],
    "device": ['cuda'],#["cpu", "cuda"],
    "includes_loading": [False],#[True, False],
    "batch_size": [128]#, 8, 16, 32, 64, 128],
}

for p in timer_.iter(params):

    if p["model_cls"] == YunetDetector and p["device"] == "cuda":
        continue

    model = p["model_cls"](device=p["device"])

    if p["includes_loading"]:
        img = get_example_image(load_numpy=False, load_torch=False)
        img = [img] * p["batch_size"]
    else:
        img = get_example_image(load_torch=True, device=p["device"])
        img = img.repeat(p["batch_size"], 1, 1, 1)

    timer_.time(model, [img], n_warmup=3, repeats=20, params=p)

df = timer_.to_df()
print(df.groupby(list(params.keys())).mean())
f_out = Path(__file__).parent / "speed_benchmark_detect.tsv"
df.to_csv(f_out, sep="\t", index=False)

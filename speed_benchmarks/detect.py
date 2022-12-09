from pathlib import Path
from medusa.data import get_example_frame
from medusa.detect import RetinanetDetector, YunetDetector

from timer import FancyTimer

timer_ = FancyTimer()
params = {
    'model_cls': [RetinanetDetector, YunetDetector],
    'device': ['cpu', 'cuda'],
    'includes_loading': [True, False],
    'batch_size': [1, 2, 4, 8, 16, 32, 64, 128]
}

for p in timer_.iter(params):

    if p['model_cls'] == YunetDetector and p['device'] == 'cuda':
        continue

    model = p['model_cls'](device=p['device'])

    if p['includes_loading']:
        img = get_example_frame(load_numpy=False, load_torch=False)
        img = [img] * p['batch_size']
    else:
        img = get_example_frame(load_torch=True, device=p['device'])
        img = img.repeat(p['batch_size'], 1, 1, 1)

    timer_.time(model, [img], n_warmup=2, repeats=10, params=p)

df = timer_.to_df()
f_out = Path(__file__).parent / 'speed_benchmark_detec.tsv'
df.to_csv(f_out, sep='\t', index=False)

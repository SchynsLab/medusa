from timer import FancyTimer

from medusa.data import get_example_video
from medusa.io import VideoLoader

timer_ = FancyTimer()
vid = get_example_video()

params = {
    'device': ['cpu', 'cuda'],
    'batch_size': [1, 16, 32, 64, 128]
}

def f(batch_size, device):
    loader = VideoLoader(vid, batch_size=batch_size, device=device)

    for batch in loader:
        pass

for p in timer_.iter(params):
    timer_.time(f, [p['batch_size'], p['device']], n_warmup=1, repeats=10, params=p)

df = timer_.to_df()
print(df.groupby(list(params.keys())).mean())

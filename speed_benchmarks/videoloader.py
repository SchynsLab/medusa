from medusa.io import VideoLoader
from medusa.data import get_example_video
from timer import FancyTimer


timer_ = FancyTimer()
vid = get_example_video()

def f():
    loader = VideoLoader(vid, batch_size=32, device='cpu')
    for _ in loader:
        pass

timer_.time(f, [], n_warmup=1, repeats=10)
df = timer_.to_df()
print(df)
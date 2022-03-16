import os
import sys
import numpy as np
import pandas as pd
from glob import glob


folder = sys.argv[1]
files = sorted(glob(os.path.join(folder, 'raw', '*.png')))
ft = [f.split('time-')[1].split('.')[0] for f in files]
ft = np.array([int(t) / 1000. for t in ft])
df = pd.DataFrame(ft, columns=['frame_times'])
df.to_csv(os.path.join(folder, 'frame_times.tsv'), sep='\t', index=False)
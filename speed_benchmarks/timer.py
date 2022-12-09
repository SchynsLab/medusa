import time
import pandas as pd
from tqdm import tqdm
from functools import reduce
from itertools import product
from collections import defaultdict


class FancyTimer:
    """Fancy timer to time Python functions."""
    def __init__(self):
        self._timers = defaultdict(list)

    @staticmethod
    def iter(params, pbar=True):
        keys = list(params)
        to_iter = product(*map(params.get, keys))

        if pbar:
            n_iters = reduce(lambda x, y: x * y, map(len, params.values()))
            to_iter = tqdm(to_iter, total=n_iters)

        for values in to_iter:
            yield dict(zip(keys, values))

    def time(self, f, args, n_warmup=2, repeats=10, params=None):
        """Time a function for a number of repetitions.

        Parameters
        ----------
        f : func
            Function to time
        args : iterable
            Arguments to pass to ``f``
        n_warmup : int
            Number of warmup calls to ``f``
        repeats : int
            Number of times the function ``f`` is called
        params : dict
            Extra parameters to add to output dictionary

        Examples
        --------
        >>> timer = FancyTimer()
        >>> timer.time(lambda x: x ** 2, [5])
        >>> timer
        """
        for _ in range(n_warmup):
            _ = f(*args)

        for i in range(repeats):
            start = time.perf_counter()
            _ = f(*args)
            dur = time.perf_counter() - start
            self._timers['duration'].append(dur)

            if params is not None:
                for k, v in params.items():
                    if type(v) == type:
                        v = v.__str__(v)
                    self._timers[k].append(v)

    def to_df(self):
        df = pd.DataFrame(self._timers)
        return df

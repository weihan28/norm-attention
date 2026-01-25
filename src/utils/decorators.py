import time
from functools import wraps

import torch

# temporary global seed
SEED = 42


class Timed:
    def __init__(self, tag=None):
        self.tag = tag

    def __call__(self, fn):
        name = self.tag or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                output = fn(*args, **kwargs)
                print(f"{name}: {time.perf_counter() - start:.6f}s")
                return output
            except Exception as e:
                print(f"{name}: Something went wrong.")
                raise

        return wrapper


def set_seed(seed):
    torch.manual_seed(seed)


class SetSeed:
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else SEED

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            set_seed(self.seed)
            return fn(*args, **kwargs)

        return wrapper

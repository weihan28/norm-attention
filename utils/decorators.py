import time
from functools import wraps


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

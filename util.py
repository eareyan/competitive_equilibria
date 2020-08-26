import time
from functools import wraps


def timing(f, before_message):
    @wraps(f)
    def wrap(*args, **kw):
        print(f"{before_message}", end='')
        t0 = time.time()
        result = f(*args, **kw)
        print(f"\t -> done, it took {time.time() - t0 : .4f}s")
        return result

    return wrap

import json
import time
import zipfile
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


def read_json_from_zip(json_world_loc):
    """
    Reads a JSON file from a zip file.
    """
    with zipfile.ZipFile(json_world_loc, "r") as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = json.loads(f.read().decode("utf-8"))
    return data

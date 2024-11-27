from pathlib import Path


def path(*args):
    DATA_PREFIX = Path(__file__).parent.parent.joinpath("_data")
    return DATA_PREFIX.joinpath(*args)

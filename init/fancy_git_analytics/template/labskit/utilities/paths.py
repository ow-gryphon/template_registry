from pathlib import Path


def get_path(path: Path):
    """
    Establishes that a given path exists
    :param path: Pathlib path to get
    :return: the path
    """
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    return path

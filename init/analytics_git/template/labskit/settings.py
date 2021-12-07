import os
from pathlib import Path

from labskit.utilities.paths import get_path


class Settings:
    """
    Project settings to track directories and settings file configurations
    """
    def __init__(self, root_path=None):
        self.root_path = Path(root_path) if root_path is not None else Path(os.getcwd())

    @property
    def data(self):
        return get_path(self.root_path / "data")

    @property
    def raw_data(self):
        return get_path(self.data / "raw_data")

    @property
    def log_dir(self):
        return get_path(self.root_path / "log")

    @property
    def output(self):
        return get_path(self.root_path / "outputs")

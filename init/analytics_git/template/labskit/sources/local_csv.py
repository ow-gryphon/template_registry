import pandas as pd

from labskit.settings import Settings


class LocalCsv:
    """
    Source for data as pandas data frames.

    :param settings: Labskit Project Settings
    :param filename: The name of the file in the `raw_data` folder to read.
    """
    def __init__(self, settings: Settings, filename):
        self.settings = settings
        self._data = None
        self.filename = filename

    @property
    def data(self):
        """
        Call to read the underlying dataset as a pandas dataframe
        :return: Loaded CSV data
        """
        if self._data is None:
            self._data = pd.read_csv(self.settings.raw_data / self.filename)
        return self._data

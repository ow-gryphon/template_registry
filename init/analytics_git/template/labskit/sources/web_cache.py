import abc
import logging

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from labskit.settings import Settings


class WebCsv(abc.ABC):
    """
    Local caching operator for downloading a datafile and persisting it locally using Parquet
     for storage.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self._data = None

    @property
    @abc.abstractmethod
    def simple_name(self):
        """
        Defines a simplified name for this data collection. Any parameters that specify metadata
        to uniquely identify this slice of data should be part of the simple name.
        :return: string
        """
        pass

    @property
    @abc.abstractmethod
    def url(self):
        """
        Defines the url where the data can be downloaded
        :return:  URL String
        """
        pass

    @property
    def data(self):
        """
        Caching call to get the underlying dataset
        :return: Cached CSV Dataset
        """
        if self._data is None:
            local_log = logging.getLogger(__name__)
            cache_file = (self.settings.raw_data / self.simple_name).with_suffix('.parquet')
            if not cache_file.exists():
                local_log.info("Downloading {} from web".format(self.simple_name))
                direct = pd.read_csv(self.url)
                table = pa.Table.from_pandas(direct)
                pq.write_table(table, cache_file)
                self._data = direct

            else:
                local_log.info("Reading {} from cache".format(self.simple_name))
                parquet = pq.read_table(cache_file)
                self._data = parquet.to_pandas()

        return self._data

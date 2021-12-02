# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from labskit import Settings
from src.data_loading.mmc_stock import LocalMMC
from src.data_processing.example_data_processing import create_timeseries


class TimeseriesIndexData(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.input_data = LocalMMC(self.settings).data

    def test_transformation_timeseries(self):
        time_series = create_timeseries(self.input_data)
        self.assertIsInstance(time_series.index, pd.DatetimeIndex)

# -*- coding: utf-8 -*-
import unittest

import numpy as np

from labskit import Settings
from src.analysis.example_analysis import get_rolling_mean
from src.data_loading.mmc_stock import LocalMMC


class RollingMeanWindow(unittest.TestCase):

    def setUp(self):
        self.settings = Settings()
        self.input_data = LocalMMC(self.settings).data.iloc[:20, :]

    def test_window_is_nan(self):
        with self.assertRaises(ValueError):
            get_rolling_mean(self.input_data,
                             window=np.nan, column='close')

    def test_window_is_zero(self):
        rolling_mean = get_rolling_mean(self.input_data,
                                        window=0, column='close')
        null_values = rolling_mean.isnull().sum()
        self.assertEqual(null_values, 20)

    def test_window_rolling(self):
        rolling_mean = get_rolling_mean(self.input_data,
                                        window=3, column='close')
        self.assertTrue(rolling_mean[0], np.nan)
        self.assertTrue(rolling_mean[1], np.nan)
        self.assertAlmostEqual(rolling_mean[2], 84.236666333333332)
        self.assertAlmostEqual(rolling_mean[19], 83.216666999999958)

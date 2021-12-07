# -*- coding: utf-8 -*-
import unittest

from labskit import Settings
from src.data_loading.mmc_stock import LocalMMC


class ParseInputData(unittest.TestCase):

    def setUp(self):
        self.settings = Settings()
        self.input_data = LocalMMC(self.settings).data.iloc[:20, :]

    def test_read_csv_data_headers(self):
        input_data_headers = [
            'date', 'open', 'high', 'low', 'close',
            'adjClose', 'volume']
        self.assertEqual(
            self.input_data.columns.tolist(),
            input_data_headers
        )
        self.assertTupleEqual(self.input_data.shape, (20, 7))

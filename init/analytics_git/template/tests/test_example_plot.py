# -*- config: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src.reporting.timeseries_plot import share_price


class ExamplePlot(unittest.TestCase):

    def setUp(self):
        self.input_data = pd.DataFrame(
            {'Column1': [1, 2, 3, 4, 5],
             'Column2': [6, 7, 8, 9, 5]})

    def test_file_is_written(self):
        with tempfile.NamedTemporaryFile() as file_pointer:
            share_price(self.input_data, ['Column1'], file_pointer)
            outpath = Path(file_pointer.name)
            self.assertTrue(outpath.exists())
            # We should write more than 100 bytes of plot
            self.assertGreater(outpath.stat().st_size, 100)

    def test_plot_one_column_axes(self):
        ax = share_price(self.input_data, ['Column1'])
        self.assertEqual(len(ax.lines), 1)

    def test_plot_one_column_series(self):
        ax = share_price(self.input_data, ['Column1'])
        xy_data = ax.lines[0].get_xydata().T
        index_array = np.array(self.input_data.index.tolist())
        column1_array = np.array(self.input_data['Column1'])
        np.testing.assert_allclose(xy_data[0], index_array)
        np.testing.assert_allclose(xy_data[1], column1_array)

    def test_plot_no_columns_dataframe(self):
        ax = share_price(self.input_data)
        xy_data_column1 = ax.lines[0].get_xydata().T
        xy_data_column2 = ax.lines[1].get_xydata().T
        index_array = np.array(self.input_data.index.tolist())

        column1_array = np.array(self.input_data['Column1'])
        np.testing.assert_allclose(xy_data_column1[0], index_array)
        np.testing.assert_allclose(xy_data_column1[1], column1_array)

        column2_array = np.array(self.input_data['Column2'])
        np.testing.assert_allclose(xy_data_column2[0], index_array)
        np.testing.assert_allclose(xy_data_column2[1], column2_array)

        self.assertEqual(len(ax.lines), 2)

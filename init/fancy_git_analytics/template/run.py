import labskit
from labskit.storage import SessionStore

from src.analysis.example_analysis import get_rolling_mean
from src.data_loading import mmc_stock
from src.data_processing.example_data_processing import create_timeseries
from src.reporting.timeseries_plot import share_price


@labskit.app
def mmc_stock_analysis(settings, outputs: SessionStore):
    """
    Executes a basic data flow to compute and plot the rolling 10 day mean of the MMC stock price.
    The output is a plot saved to a file named "mmc_series.png"
    """
    timeseries_data = mmc_stock.LocalMMC(settings).data \
        .assign(rolling_mean=lambda x: get_rolling_mean(x, window=10, column='close'))\
        .pipe(create_timeseries)

    share_price(
        timeseries_data,
        columns=['close', 'rolling_mean'],
        output_file_path=outputs.project_file('mmc_series.png'))


if __name__ == '__main__':
    mmc_stock_analysis()

import logging

from matplotlib import pyplot as plt


def share_price(data, columns=None, output_file_path=None):
    """
    Plot a timeseries graph with simple axis formatting.

    :param data: The dataframe to plot
    :param columns: List of column names to plot
    :param output_file_path: Path location to save figure
    :return: matplotlib axis object
    """
    logger = logging.getLogger(__name__)

    fig, ax = plt.subplots()
    x = data.index
    try:
        for column in columns:
            ax.plot(x, data[column], label=column)
        ax.legend()
    except TypeError:
        logger.error('No columns passed, the plot may not be reliable')
        ax.plot(x, data)
    fig.autofmt_xdate()
    ax.set_xlabel('Time')
    ax.set_ylabel('Share price')

    if output_file_path is not None:
        fig.savefig(output_file_path)
    return ax

def get_rolling_mean(dataframe, window, column):
    """
    Calculate the rolling mean for a given column.

    :param dataframe: dataframe to find rolling means
    :param window: the window specification as an integer in time periods
    :param column: the name of the data column to average

    :return: rolling mean of the given column
    """

    if not isinstance(window, int):
        raise ValueError('The window value be an integer')

    rolling_mean = dataframe[column].rolling(window).mean()
    return rolling_mean

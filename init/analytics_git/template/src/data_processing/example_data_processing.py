import pandas as pd


def create_timeseries(dataframe):
    """
    Convert the column 'Date' to datetime format and set
    as index.

    :param dataframe: incoming data to transform which contains
    the 'date' column
    :type dataframe: pd.DataFrame

    :return: dataframe with the Date column transformed
    into a datetime object and set as index.
    """
    return dataframe.assign(date_formatted=lambda x: pd.to_datetime(x['date']))\
        .set_index('date_formatted')\
        .drop('date', axis=1)

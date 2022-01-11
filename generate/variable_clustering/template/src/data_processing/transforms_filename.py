import pandas as pd
import numpy as np
from functools import partial
import logging


def define_problem(raw_frame):
    """
    Cleans the input, raw data and defines the set of columns that should be
    used for modeling

    :param raw_frame: the raw HMDA dataset
    :return: A dataframe clean and fit for clustering
    """

    selected_columns = [
        'applicant_income_000s',
        'loan_purpose_name',
        'hud_median_family_income',
        'loan_amount_000s',
        'originated'
    ]
    log = logging.getLogger(__name__)
    log.info("Input data size {}".format(raw_frame.shape))

    return raw_frame.assign(
        originated=lambda f: (f['action_taken'] == 1).astype('int'))\
        .loc[:, selected_columns]


def clean_clustering_data(frame):
    """
    Cleans the raw input DataFrame by expanding categorical variables, filtering NaN values,
    and adding several features that correspond to log values of the original features.

    :param frame: A Pandas DataFrame.
    :return: A cleaned Pandas DataFrame.
    """

    features_to_log = ['hud_median_family_income', 'loan_amount_000s']

    # Add Categorical Variables
    expanded_frame = pd.get_dummies(frame, 'loan_purpose_name')

    # Filter NaNs
    expanded_filtered_frame = expanded_frame.loc[lambda f: ~np.any(f.isnull(), axis=1)]

    def log_single(frame, feature):
        return np.log(frame[feature])

    # Add Log values of features.
    new_columns = {
        '{feature}_log'.format(feature=feat): partial(log_single, feature=feat)
        for feat in features_to_log
    }

    cleaned_frame_w_logs = expanded_filtered_frame.assign(**new_columns)

    return cleaned_frame_w_logs

from sklearn import cluster
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def train_predict_clustering_model(frame, target):
    """
    Fits a clustering model on a cleaned DataFrame.  Predicts cluster membership using
    the fitted model.

    :param frame: A Pandas DataFrame containing both independent and dependent variables.
    :param target: The name of the dependent variable.
    :return: The frame after .predict is called from K-Means,
    which includes the cluster memberships.
    """

    cluster_model = cluster.KMeans(n_clusters=2)

    x_values_from_frame = frame.loc[:, frame.columns != target]

    x_train, x_test = train_test_split(x_values_from_frame)

    fit_model = cluster_model.fit(x_train)

    predicted_model = pd.DataFrame(np.array(fit_model.predict(x_test)),
                                   columns=['originated_prediction'])
    with_prediction = pd.concat([x_test.reset_index(), predicted_model], axis=1)

    return with_prediction

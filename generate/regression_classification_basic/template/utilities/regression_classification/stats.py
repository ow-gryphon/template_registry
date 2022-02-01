# Basic statistical tests and metrics

import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.stats
from statistics import variance
from typing import Any, List, Union, Callable, Tuple, Optional


def accuracy_score(
    y_true: List,
    y_pred: List,
    normalize: bool = True,
    sample_weight: Optional[List] = None,
) -> float:
    """
    Accuracy as measured between true and predicted values for columns in lists

    Uses metrics.accuracy_score

    :param y_true: a list of true values
    :param y_pred: a list of predicted values
    :param normalize: fraction/number flag: True if fraction of correct desired, False if number
        of correct desired [optional, default is True]
    :param sample_weight: a list of sample weights [optional, default is None]

    :return: accuracy score, as a float
    """
    return metrics.accuracy_score(y_true, y_pred, normalize = normalize, sample_weight = sample_weight)


def roc(
    y_true: List,
    y_score: List,
    pos_label: Optional[Union[int, str]] = None,
    sample_weight: Optional[List] = None,
    drop_intermediate: bool = False,
) -> Any:
    """
    Calculates False Positive Rate (FPR), True Positive Rate (TPR)
    and thresholds on the decision function for FPR and TPR using sklearn
    implementation.

    :param y_true: array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1},
        then pos_label should be explicitly given.

    :param y_score: array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    :param pos_label: int or str, [Optional, default=None]
        Label considered as positive and others are considered negative.

    :param sample_weight: array-like of shape = [n_samples], [Optional, default = None]
        Sample weights.

    :param drop_intermediate: boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve

    :return: fpr -- array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
        
        tpr -- array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
        
        thresholds -- array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being
        predicted and is arbitrarily set to `max(y_score) + 1`.
    """
    return metrics.roc_curve(
        y_true, y_score, pos_label, sample_weight, drop_intermediate
    )


def auc(x: List, y: List) -> float:
    """
    Calculates the area under a curve using the trapezoidal rule

    Uses metrics.auc

    :param x: a list of x coordinates. These must be either monotonic increasing or monotonic
        decreasing
    :param y: a list of y coordinates

    :return: AUC score, as a float
    """
    return metrics.auc(x, y)


def ks(
    rvs: Union[str, Union[List, Callable]],
    cdf: Union[str, Callable],
    args: Any = (),
    N: int = 20,
    alternative: str = "two-sided",
    mode: str = "approx",
) -> Tuple[float, float]:
    """
    KS test to test the degree to which two distributions differ

    Uses scipy.stats.kstest

    :param rvs: (str, array or callable) the known distribution, data or callable distribution
    :param cdf: (str or callable) the string of known distribution in scipy.stats or callable
        distribution
    :param args: (tuple or sequence) if rvs or cdf are strings, this is a distribution parameter
        e.g. loc and scale for a 'norm' distribution [optional]
    :param N: the sample size [optional, default is 20]
    :param alternative: ('two-sided', 'less', 'greater') corresponds to the type of alternative
        hypothesis [optional, default='two-sided']
    :param mode: ('approx' or 'asymp') determines how the p-value is estimated
        [optional, default='approx']

    :return: 
        KS test statistic, as a float
        
        approximate p-value, as a float
    """
    return scipy.stats.kstest(rvs, cdf, args, N, alternative, mode)


def gini(
    y_true: List,
    y_score: Union[List[float], List[List[float]]],
    average: str = "macro",
    sample_weight: Optional[List] = None,
    invert: bool = False,
    confidence_interval: Optional[Callable] = None,
    level: float = 0.95,
    **kwargs: Any,
) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Calculates the Gini coefficient and optional confidence intervals from the ROC curve

    Uses metrics.roc_auc_curve

    :param y_true: an array of true values,
    :param y_score: an array of shape [n_samples] or [n_samples, n_classes]
        These scores are the target scores; either are the probability estimates of the positive
        class, confidence values or non-threshold measure of decisions
    :param average: a string to determine the type of averaging performed on the data
        [optional, default is 'macro']
        None: the scores for each class are returned. If none is not assigned, this parameter
        determines the type of averaging to be performed

        micro:
            calculates the metrics globally by considering each element of the label indicator
            matrix as a label i.e. total true positives and total false positives

        macro:
            calculates the metrics for each label and finds the unweighted mean. Does not take
            into account label imbalance

        weighted:
            calculates the metrics for each label, then finds the average where each metric
            is labeled by the support (number of true instances for each label)

        samples:
            calculates metrics for each instance, and finds the average

    :param sample_weight: an array of shape [n_samples] to determine how each sample is weighted in
        calculations, [optional, default is None]
    :param invert: a boolean value which determines which form of the Gini from ROC calculation to use.
        if True, :math:`G = 1 - (2 AUC)`. if False, :math:`G = (2 AUC) - 1`
        [optional, default is False]
    :param confidence_interval: a callable to indicate which confidence interval function
        is to be used. [optional, default is None]
        Any method which has API following (y_true, y_score, gini, auc, level))
    :param level: a float indicating the confidence level used in the confidence_interval.
        [optional, default is 0.95]
    :param kwargs: a dictionary containing any optional parameters that may be required for the
        confidence_interval.

    :return: 
        Gini coefficient, as a float

        lower and upper bound confidence interval if confidence interval function is selected. These
        are returned with the gini coefficient as a tuple of the format (gini, (lower CI, upper CI)).
    """
    auc = metrics.roc_auc_score(y_true, y_score, average=average, sample_weight=sample_weight)

    result = (2 * auc) - 1

    if invert == True:
        result = -1 * result

    if confidence_interval:
        try:
            result = (
                result,
                confidence_interval(y_true, y_score, result, auc, level, **kwargs),
            )

        except Exception as e:
            raise Exception(
                f"Error in confidence interval used in gini function. The error in the confidence interval function is '{e}'"
            )

    return result


def r_squared_mcfadden(likelihood_fitted: float, likelihood_null: float) -> float:
    """
    Computes the McFadden r squared statistic

    :param likelihood_fitted: a float of the (maximised) likelihood value from the current
        fitted model
    :param likelihood_null: a float of the corresponding value of the likelihood from the
        null model

    return: McFadden r squared statistic, as a float
    """
    if likelihood_fitted <= 0 or likelihood_null <= 0:
        raise ValueError("parameters need to be positive")

    return 1 - (math.log(likelihood_fitted) / math.log(likelihood_null))


def precision_recall_plot(y_true: List, y_pred: List):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.figure()
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='Random')
    plt.plot(recall, precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    return plt.gcf()


def roc_plot(y_true: List, y_pred: List):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot([0,1], [0,1], linestyle='--', label='Random')
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    return plt.gcf()


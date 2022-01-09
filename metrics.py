from typing import Dict, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import sklearn
from sklearn.base import BaseEstimator

from config import ACCEPTABLE_RECALL

#### METRICS ####


def get_threshold_and_precision_at_recall(
    truth: Iterable[float],
    scores: Iterable[float],
    acceptable_recall: float = ACCEPTABLE_RECALL,
) -> Tuple[float, float]:
    """Compute the precision at a given recall (specified by acceptable_recall)"""
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        truth, scores
    )

    # Use root_scalar to find when recall is at acceptable_recall
    function_to_minimize = interp1d(thresholds, recall[:-1] - acceptable_recall)
    threshold = root_scalar(
        function_to_minimize, bracket=(thresholds[0], thresholds[-1])
    ).root

    precision_at_recall = float(interp1d(thresholds, precision[:-1])(threshold))
    return threshold, precision_at_recall


def get_scores(X: Union[pd.DataFrame, np.ndarray], model: BaseEstimator) -> np.ndarray:
    """Compute the probability scores for a given model on data X"""
    try:
        scores = model.predict_proba(X)[:, 1]
    except AttributeError:
        scores = model.decision_function(X)
    return scores


def get_precision_and_roc_auc(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model: BaseEstimator,
) -> Dict[str, float]:
    """Compute roc auc and precision at given recall for model on data X and label y"""
    scores = get_scores(X, model)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y, scores)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    _, precision = get_threshold_and_precision_at_recall(y, scores)
    return {
        f"precision_at_{100*ACCEPTABLE_RECALL}_recall": precision,
        "roc_auc": roc_auc,
    }


def get_confusion_status(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    model: BaseEstimator,
) -> pd.DataFrame:
    """Add status column to dataframe X containing whether a row is a True Positive (TP), True Negative (TN)
    False Positive (FP) or False Negative (FN)"""
    X = X.copy()
    scores = get_scores(X, model)
    threshold = get_threshold_and_precision_at_recall(y, scores)[0]
    X["proba"] = scores
    X["prediction"] = scores >= threshold
    X["status"] = "TP"
    X.loc[
        lambda df: (~df["prediction"].astype(bool)) & (y.astype(bool)), "status"
    ] = "FN"
    X.loc[
        lambda df: (~df["prediction"].astype(bool)) & (~y.astype(bool)), "status"
    ] = "TN"
    X.loc[
        lambda df: (df["prediction"].astype(bool)) & (~y.astype(bool)), "status"
    ] = "FP"
    return X

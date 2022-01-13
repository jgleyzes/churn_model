from typing import Dict, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import sklearn
from sklearn.base import BaseEstimator

from metrics import get_scores, get_threshold_and_precision_at_recall


def plot_correlations(df: pd.DataFrame):
    """Plot correlations between all columns in dataframe"""
    plt.figure(figsize=(16, 9))
    Var_Corr = df.corr()
    sns.heatmap(
        Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True
    )


def plot_importance(importances: Iterable[float], feature_names: Iterable[str]):
    """Plot horizontal bar chart with importances for features"""
    x, y = np.array([importances, feature_names])[:, np.argsort(importances)]
    fig = go.Figure(go.Bar(x=x.astype(float), y=y, orientation="h"))

    fig.show()


def _get_scatter_precision_recall(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model: BaseEstimator,
    name: str = None,
):
    """Compute precision and recall for model on X, y and prepare for plot"""
    scores = get_scores(X, model)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y, scores)

    return go.Scatter(x=recall, y=precision, hovertext=thresholds, name=name)


def plot_precision_recall(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model: BaseEstimator,
):
    """Plot precision and recall for a single model on X, y"""

    fig = go.Figure(data=_get_scatter_precision_recall(X, y, model))
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    fig.show()


def plot_precision_recall_multiples(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    dict_models: Dict[str, BaseEstimator],
):
    """Plot precision and recall for a multiple models on X, y"""

    fig = go.Figure(
        data=[
            _get_scatter_precision_recall(X, y, model, name)
            for name, model in dict_models.items()
        ]
    )
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    fig.show()


def plot_confusion_matrix(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model: BaseEstimator,
):
    """Compute and plot confusion matrix for model on X, y"""
    scores = get_scores(X, model)
    threshold, _ = get_threshold_and_precision_at_recall(y, scores)
    predictions = scores >= threshold
    cm = sklearn.metrics.confusion_matrix(y, predictions, labels=[0, 1])
    disp = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[0, 1]
    )
    disp.plot()

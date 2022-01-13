from collections import defaultdict
from copy import copy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import base
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

from metrics import get_precision_and_roc_auc


class ModelComparator:
    """
    Compare models by train and evaluating on n_tries different realizations of the train test split.
    The goal is to see how often a given model performs better than another in a more statistically robust way
    than just comparing on a single test set.
    """

    def __init__(self, models_dict: Dict[str, BaseEstimator], n_tries: int = 25):

        self.models_dict = models_dict
        self.n_tries = n_tries

    @staticmethod
    def _compute_scores_model_and_split(
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[float, float]:
        model = copy(model)
        model.fit(X_train, y_train)
        precision, roc_auc = list(
            get_precision_and_roc_auc(X_test, y_test, model).values()
        )
        return precision, roc_auc

    @staticmethod
    def _add_best_model_columns(df: pd.DataFrame) -> pd.DataFrame:
        models = df.columns
        df["best"] = df.apply(lambda x: models[np.argmax(x)], axis=1)
        return df

    @ignore_warnings(category=ConvergenceWarning)
    def compare_models(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        auc_scores = defaultdict(list)
        precision_scores = defaultdict(list)
        for random_state in tqdm(range(self.n_tries)):
            for name, model in self.models_dict.items():
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, stratify=y, random_state=random_state
                )
                precision, roc_auc = self._compute_scores_model_and_split(
                    model, X_train, y_train, X_test, y_test
                )
                auc_scores[name].append(roc_auc)
                precision_scores[name].append(precision)
        auc_scores = self._add_best_model_columns(pd.DataFrame(auc_scores))
        precision_scores = self._add_best_model_columns(pd.DataFrame(precision_scores))
        return precision_scores, auc_scores

    @staticmethod
    def compare_score_to_baseline(
        scores_df: pd.DataFrame, baseline_model_name: str, direction: str = "maximize"
    ) -> pd.DataFrame:
        comparison_summary = {}
        for col in scores_df:
            if col not in ["best", baseline_model_name]:
                change_in_score = (
                    (scores_df[col] - scores_df[baseline_model_name])
                    / scores_df[baseline_model_name]
                ).mean()
                ratio_of_wins = (scores_df[col] > scores_df[baseline_model_name]).mean()
                if direction == "minimize":
                    ratio_of_wins = 1 - ratio_of_wins
                comparison_summary[col] = {
                    "change_in_score": change_in_score,
                    "ratio_of_wins": ratio_of_wins,
                }
        return pd.DataFrame(comparison_summary)

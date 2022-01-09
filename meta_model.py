from typing import Dict, Union
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from scipy.interpolate import interp1d

import metrics


class MetaModel:
    def __init__(self, models_dict: Dict[str, BaseEstimator]) -> None:
        self.models_dict = models_dict
        self.mappings_to_precision: Dict[str, interp1d] = {}

    @staticmethod
    def _get_mapping_to_precision(
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model: BaseEstimator,
    ) -> interp1d:
        """Compute the threshold corresponding to the ACCEPTABLE RECALL"""
        scores = metrics.get_scores(X, model)
        precisions, _, thresholds = sklearn.metrics.precision_recall_curve(y, scores)
        score_to_precision = interp1d(
            thresholds, precisions[:-1], bounds_error=False, fill_value=(0, 1)
        )
        return score_to_precision

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ):
        """Run the fit method on all the submodels, and compute thresholds at 90% recall"""
        for name, model in self.models_dict.items():
            self.models_dict[name] = model.fit(X, y)
            # The threshold is computed on the training data, not on the test data
            # otherwise that would be cheating (we can't compute a threshold on the fly when doing inference
            # it needs to have been computed beforehand)
            self.mappings_to_precision[name] = self._get_mapping_to_precision(
                X, y, model
            )
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict using the voting rule (if more than half of submodels predict True, than True)"""
        preds = []
        for model in self.models_dict.values():
            preds.append(model.predict(X))
        return (np.mean(preds, axis=0) > 0.5).astype(int)

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get decision scores by taking the average of the submodels scores rescaled by their precomputed thresholds"""
        all_scores = []
        for name, model in self.models_dict.items():
            scores = metrics.get_scores(X, model)
            # Normalize scores by the threshold at ACCEPTABLE_RECALL
            all_scores.append([self.mappings_to_precision[name](s) for s in scores])
        raw_scores = np.mean(all_scores, axis=0)
        return raw_scores

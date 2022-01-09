from typing import Callable, Iterable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


class ModelRecursiveFeatureElimination:
    """This wrapper around sklearn's RFECV allows to run the recursive feature elimination on a given model and
    then produce a pipeline which selects the right columns before applying to the chosen model"""

    cv = StratifiedKFold(3)
    scoring = "roc_auc"
    min_features_to_select = 1
    step = 1

    def _get_column_selector(
        self, all_columns: np.ndarray, rfecv: RFECV
    ) -> ColumnTransformer:

        self.cols_to_keep = list(all_columns[rfecv.ranking_ == 1])
        column_selector = ColumnTransformer(
            [("selector", "passthrough", self.cols_to_keep)], remainder="drop"
        )
        return column_selector

    def _recursive_feature_elimination(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], clf: BaseEstimator
    ) -> RFECV:
        """Apply the RFECV per say"""
        rfecv = RFECV(
            estimator=clf,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=self.min_features_to_select,
            importance_getter="auto",
        )
        rfecv.fit(X, y)
        return rfecv

    def get_model(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], clf: BaseEstimator
    ) -> Pipeline:
        """Apply RFE and then create pipeline with column selector first and then clf"""
        rfecv = self._recursive_feature_elimination(X, y, clf)
        column_selector = self._get_column_selector(X.columns, rfecv)
        model = Pipeline(
            [("column_selector", column_selector), ("clf", rfecv.estimator_)]
        )
        return model


class PipelineRecursiveFeatureElimination(ModelRecursiveFeatureElimination):
    """Same as ModelRecursiveFeatureElimination but for sklearn's Pipeline"""

    def importance_getter(self, pipe: Pipeline) -> Callable:
        try:
            return pipe.named_steps.clf.coef_
        except AttributeError:
            return pipe.named_steps.clf.feature_importances_

    def get_model(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], pipe: Pipeline
    ) -> Pipeline:
        rfecv = self._recursive_feature_elimination(X, y, pipe)
        column_selector = self._get_column_selector(X.columns, rfecv)
        model = Pipeline([("column_selector", column_selector), *pipe.steps])
        return model

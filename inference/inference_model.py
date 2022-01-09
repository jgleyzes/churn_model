import pickle
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

from logger_factory import create_logger
from utils import convert_strings_to_one_hot

logger = create_logger(__name__)


class InferenceModel(ABC):
    """Generic wrapper around a sklearn type model to serve it in opyrator"""

    def __init__(self, model_path: str, threshold: float) -> None:
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.threshold = threshold

    @abstractmethod
    def _predict_proba(self, features_df: pd.DataFrame) -> float:
        pass

    def predict(self, features: Dict[str, float]) -> bool:
        """Convert features (dict) to dataframe for evaluation by the model"""
        features_df = pd.DataFrame([features])
        features_df_numerical = convert_strings_to_one_hot(features_df)
        probability = self._predict_proba(features_df_numerical)
        logger.info(
            f"Predicted a probability {probability}, for a threshold of {self.threshold}"
        )
        return bool(probability >= self.threshold)


class LGBMInferenceModel(InferenceModel):
    """InferenceModel wrapper specific to LGBMClassifier"""

    @staticmethod
    def _set_unseen_categories_to_zero(
        features_df: pd.DataFrame, expected_feature_names: List[str]
    ) -> pd.DataFrame:
        """The model expect a column per category value, so we need to fill with 0 the category values not
        corresponding to the present features_df"""
        missing_features = list(set(expected_feature_names) - set(features_df.columns))
        features_df[missing_features] = 0
        return features_df

    def _predict_proba(self, features_df: pd.DataFrame) -> float:

        full_features_df = self._set_unseen_categories_to_zero(
            features_df, self.model.feature_name_
        )
        sorted_features = full_features_df[self.model.feature_name_]
        return float(self.model.predict_proba(sorted_features)[:, 1])

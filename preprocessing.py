import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from inference.utils import convert_strings_to_one_hot


def drop_columns_and_convert_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing to make the data ML ready"""
    df = df.drop(["CustomerId", "Surname"], axis=1)

    # Use same preprocessing as inference to avoid mismatch
    df = convert_strings_to_one_hot(df)
    return df


def split_label(df: pd.DataFrame, label_column: str = "Exited") -> pd.DataFrame:
    """Split features and label"""
    return df.drop(label_column, axis=1), df[label_column]


class DivideColumns(TransformerMixin, BaseEstimator):
    """A template for a custom transformer which creates a new column by dividing col1 and col2"""

    def __init__(self, col1: str, col2: str):
        self.col1 = col1
        self.col2 = col2

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # transform X via code or additional methods
        X = X.copy()
        X[f"{self.col1}/{self.col2}"] = X[self.col1] / X[self.col2]
        return X

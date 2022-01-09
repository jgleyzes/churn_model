from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.lib.function_base import piecewise
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.utils._testing import ignore_warnings

from metrics import get_precision_and_roc_auc
from preprocessing import split_label


class HyperParameterTuner(ABC):
    """
    Abstract class to optimize hyperparameters using optuna.
    """

    n_folds = 3
    cv: BaseCrossValidator = KFold(n_folds)
    model_type: BaseEstimator = None
    DEFAULT_PARAMS: Dict[str, Any] = {}

    def __init__(self) -> None:
        self.study: optuna.Study = None

    @staticmethod
    @abstractmethod
    def _get_trial_params(trial: optuna.Trial) -> Dict[str, Any]:
        """This function will define the hyperparameter space over which we will run the search"""
        pass

    @staticmethod
    def _fit_cv_model(
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        model: BaseEstimator,
    ) -> BaseEstimator:
        """Wrapper around the model fit method to include potential refinements
        (eg early stopping, which is why the validation data is included)"""

        return model.fit(X_train, y_train)

    @classmethod
    def _instantiate_model(cls, params: Dict[str, Any]) -> BaseEstimator:
        """Wrapper around the instantiation to allow for the use of sklearn Pipeline"""
        return cls.model_type(**params)

    def _get_precision_roc_auc_on_single_fold(
        self,
        df: pd.DataFrame,
        train_index: Iterable[int],
        val_index: Iterable[int],
        model: BaseEstimator,
    ) -> Tuple[float, float]:
        """Fit the model on training data (controlled by train index) and computes precision
        and roc_auc on val data (controlled by val index)"""

        X_train, y_train = split_label(df.iloc[train_index])
        X_val, y_val = split_label(df.iloc[val_index])
        model = self._fit_cv_model(X_train, y_train, X_val, y_val, model)
        precision, roc_auc = list(
            get_precision_and_roc_auc(X_val, y_val, model).values()
        )
        return precision, roc_auc

    @staticmethod
    def _store_extra_model_info_trial_fold(
        model: BaseEstimator,
        trial: optuna.Trial,
        fold: int,
    ) -> None:
        """Store extra model specific info in the study"""
        pass

    @ignore_warnings(category=ConvergenceWarning)
    def _run_optuna(self, df: pd.DataFrame, n_trials: int) -> optuna.Study:
        """Function to actually run the hyperparameter search with optuna"""

        def objective(trial):
            params = self._get_trial_params(trial)
            params.update(self.DEFAULT_PARAMS)
            aucs = []
            precisions = []
            for fold, (train_index, val_index) in enumerate(self.cv.split(df)):
                model = self._instantiate_model(params)

                precision, roc_auc = self._get_precision_roc_auc_on_single_fold(
                    df, train_index, val_index, model
                )
                aucs.append(roc_auc)
                precisions.append(precision)
                self._store_extra_model_info_trial_fold(model, trial, fold)
            trial.set_user_attr("mean_precision", np.mean(precisions))
            trial.set_user_attr("mean_auc", np.mean(aucs))
            return np.mean(precisions)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study

    def _extract_best_HP_from_study(self, study: optuna.Study) -> Dict[str, Any]:
        """Get best hyperparameters from study"""

        return study.best_trial.params

    def get_tuned_model(self, df: pd.DataFrame, n_trials: int = 100) -> BaseEstimator:
        """Final method which takes a dataframe, optimize the model over that dataframe and then train a model with best
        hyperparameters on the dataframe. df should only contain training data (which will be further divided into train and val)"""

        self.study = self._run_optuna(df, n_trials)
        best_hyperparameters = self._extract_best_HP_from_study(self.study)
        best_hyperparameters.update(self.DEFAULT_PARAMS)
        precision = self.study.best_trial.user_attrs["mean_precision"]
        roc_auc = self.study.best_trial.user_attrs["mean_auc"]
        print("The best hyperparameters are")
        pprint(best_hyperparameters)
        print(f"with a val precision of {precision} and roc_auc of {roc_auc}")

        best_model = self._instantiate_model(best_hyperparameters)
        X, y = split_label(df)

        return best_model.fit(X, y)


class LinearSVCHyperParameterTuner(HyperParameterTuner):
    model_type = LinearSVC
    DEFAULT_PARAMS = {"dual": False}

    @classmethod
    def _instantiate_model(cls, params: Dict[str, Any]) -> Pipeline:
        """Add MinMaxScaler before instantiating the LinearSVC"""

        return Pipeline([("scaler", MinMaxScaler()), ("clf", cls.model_type(**params))])

    @staticmethod
    def _get_trial_params(trial: optuna.Trial) -> Dict[str, Any]:
        """Search for C and penalty"""

        return {
            "C": trial.suggest_float("C", 0.1, 100, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }


class KNNHyperParameterTuner(HyperParameterTuner):
    """Optimize n_neighbors parameter in KNeighborsClassifier (including a MinMaxScaler beforehand)"""

    model_type = KNeighborsClassifier

    @classmethod
    def _instantiate_model(cls, params: Dict[str, Any]) -> Pipeline:
        """Add MinMaxScaler before instantiating the KNeighborsClassifier"""

        return Pipeline([("scaler", MinMaxScaler()), ("clf", cls.model_type(**params))])

    @staticmethod
    def _get_trial_params(trial: optuna.Trial) -> Dict[str, Any]:
        """Only search n_neighbors"""

        return {"n_neighbors": trial.suggest_int("n_neighbors", 2, 100)}


class LGBMHyperParameterTuner(HyperParameterTuner):
    """Optimize LGBMClassifier.
    LGBMClassifier allows early stopping based on a eval set, which brings some complications
    in the _fit_cv_model and _extract_best_HP_from_study methods"""

    model_type = LGBMClassifier
    DEFAULT_PARAMS = {
        "verbosity": -1,
    }

    @staticmethod
    def _fit_cv_model(
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        model: BaseEstimator,
    ) -> BaseEstimator:
        """Use LBGM early stopping method"""
        model.fit(
            X_train,
            y_train,
            eval_metric="auc",
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False,
        )
        return model

    @staticmethod
    def _get_trial_params(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        }

    @staticmethod
    def _store_extra_model_info_trial_fold(
        model: BaseEstimator, trial: optuna.Trial, fold: int
    ) -> None:
        """Store the best iteration (ie the number of estimators when the early stopping applied)
        to be retrieved for the best model"""
        trial.set_user_attr(f"n_estimators_fold_{fold}", model.best_iteration_)

    @classmethod
    def _get_num_estimators(cls, study: optuna.Study) -> int:
        """Get the number estimators for the best model using the average of n_estimators in each fold of the cross validation"""
        return int(
            np.mean(
                [
                    study.best_trial.user_attrs[f"n_estimators_fold_{fold}"]
                    for fold in range(cls.n_folds)
                ]
            )
        )

    def _extract_best_HP_from_study(self, study: optuna.Study) -> Dict[str, float]:
        """Fetch the params from best trial as well as optimal number of estimators (obtained thanks to early stopping)"""
        best_trial = study.best_trial
        best_hps = best_trial.params
        best_hps["n_estimators"] = self._get_num_estimators(study)
        return best_hps

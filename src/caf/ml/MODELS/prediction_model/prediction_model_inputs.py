# -*- coding: utf-8 -*-
"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import enum
from pathlib import Path
from typing import Optional, List, Union, Any
import numpy as np
from caf.toolkit import BaseConfig
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier,
                              RandomForestRegressor,
                              ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              AdaBoostRegressor,
                              BaggingRegressor,
                              ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import (LogisticRegression,
                                  Ridge,
                                  Lasso,
                                  ElasticNet,
                                  LinearRegression)
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


class run_file_inputs(BaseConfig):
    # # # INPUT/OUTPUT PATHS # # #
    file_path: Optional[Path] = None
    folder_path: Optional[Path] = None
    output_path: Optional[Path] = None
    validation_path: Optional[Path] = None

    # # # PROCESSING INPUT DATA # # #
    target_column: Optional[str] = None
    custom_index: Optional[List[str]] = None
    column_name_to_drop_rows: Optional[List[str]] = None
    value_in_row: Optional[List[Union[str, int, float]]] = None

    # # # TRANSFORMING DATA # # #
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None
    weight_column: Optional[str] = None
    binary_prediction: Optional[tuple[int, ...]] = None
    time_series_split: Optional[str] = None
    split_size: Optional[float] = None

    # # # MODELLING # # #
    model_choice: List[Any]  # TODO should be abc method but not compatible with baseconfig?
    full_transformations: Optional[bool] = False
    cv: Optional[str] = None
    skip_feature_selection: Optional[bool] = False
    intensive_feature_selection: Optional[bool] = False

class Models(enum.Enum):
    LOGIT_REGRESSION_L1 = (LogisticRegression, {'penalty': 'l1', 'solver': 'liblinear'})
    LOGIT_REGRESSION_L2 = (LogisticRegression, {'penalty': 'l2'})
    LOGIT_REGRESSION_ELASTICNET = (LogisticRegression, {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5})
    MULTINOMIAL = (LogisticRegression, {'multi_class': 'multinomial', 'solver': 'lbfgs'})
    RANDOM_FOREST_REGRESSOR = (RandomForestRegressor, {})
    EXTRA_TREES_REGRESSOR = (ExtraTreesRegressor, {})
    GRADIENT_BOOSTING_REGRESSOR = (GradientBoostingRegressor, {})
    GRADIENT_BOOSTING_CLASS = (GradientBoostingClassifier, {})
    ADABOOST_REGRESSOR = (AdaBoostRegressor, {})
    BAGGING_REGRESSOR = (BaggingRegressor, {})
    SVR = (SVR, {})
    KNN = (KNeighborsRegressor, {})
    RIDGE = (Ridge, {})
    LASSO = (Lasso, {})
    ELASTICNET = (ElasticNet, {})
    LINEAR_REGRESSION = (LinearRegression, {})
    DECISION_TREE_REGRESSOR = (DecisionTreeRegressor, {})
    RANDOM_FOREST_CLASSIFIER = (RandomForestClassifier, {})
    EXTRA_TREES_CLASSIFIER = (ExtraTreesClassifier, {})
    DECISION_TREE_CLASSIFIER = (DecisionTreeClassifier, {})
    SVM_CLASSIFIER = (OneVsRestClassifier, {'estimator': LinearSVC()})

    def get_model(self):
        model_class, params = self.value if isinstance(self.value, tuple) else (self.value, {})
        return model_class(**params)


class ModelGrids(enum.Enum):
    RANDOM_FOREST_REGRESSOR = {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    EXTRA_TREES_REGRESSOR = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    GRADIENT_BOOSTING_REGRESSOR = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
    }

    ADABOOST_REGRESSOR = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}

    BAGGING_REGRESSOR = {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 0.7, 1.0],
        "max_features": [0.5, 0.7, 1.0],
    }

    SVR = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "poly", "rbf"],
        "epsilon": [0.1, 0.2, 0.3],
    }

    KNN = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"], "p": [1, 2]}

    RIDGE = {"alpha": [0.1, 1.0, 10.0]}

    LASSO = {"alpha": [0.1, 1.0, 10.0]}

    ELASTICNET = {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}

    DECISION_TREE_REGRESSOR = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    LOGIT_REGRESSION_L1 = {"C": [1.0, 0.1, 0.01, 0.001]}
    LOGIT_REGRESSION_L2 = {"C": [1.0, 0.1, 0.01, 0.001]}
    LOGIT_REGRESSION_ELASTICNET = {"C": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}

    MULTINOMIAL = {"C": np.arange(0.1, 10, 0.1).tolist(), "solver": ["lbfgs", "newton-cg", "sag", "saga"]}

    RANDOM_FOREST_CLASSIFIER = {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    EXTRA_TREES_CLASSIFIER = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    DECISION_TREE_CLASSIFIER = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }

    GRADIENT_BOOSTING_CLASSIFIER = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7]
        }

    SVM_CLASSIFIER = {
            'estimator__C': [0.1, 1, 10],
            'estimator__loss': ['hinge', 'squared_hinge'],
        }

    @classmethod
    def get_grid(cls, model_enum):
        return getattr(cls, model_enum.name).value


model_instance_to_enum = {
    RandomForestRegressor: ModelGrids.RANDOM_FOREST_REGRESSOR,
    ExtraTreesRegressor: ModelGrids.EXTRA_TREES_REGRESSOR,
    GradientBoostingRegressor: ModelGrids.GRADIENT_BOOSTING_REGRESSOR,
    AdaBoostRegressor: ModelGrids.ADABOOST_REGRESSOR,
    BaggingRegressor: ModelGrids.BAGGING_REGRESSOR,
    SVR: ModelGrids.SVR,
    KNeighborsRegressor: ModelGrids.KNN,
    Ridge: ModelGrids.RIDGE,
    Lasso: ModelGrids.LASSO,
    ElasticNet: ModelGrids.ELASTICNET,
    DecisionTreeRegressor: ModelGrids.DECISION_TREE_REGRESSOR,

    RandomForestClassifier: ModelGrids.RANDOM_FOREST_CLASSIFIER,
    ExtraTreesClassifier: ModelGrids.EXTRA_TREES_CLASSIFIER,
    DecisionTreeClassifier: ModelGrids.DECISION_TREE_CLASSIFIER,
    GradientBoostingClassifier: ModelGrids.GRADIENT_BOOSTING_CLASSIFIER}

def get_model_grid(model_instance):
    model_enum = model_instance_to_enum.get(type(model_instance))
    if model_enum:
        return model_enum.value
    else:
        raise ValueError(f"No grid found for model instance of type {type(model_instance)}")

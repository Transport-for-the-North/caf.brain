# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
# Built-Ins
import enum
from pathlib import Path
from typing import Any, List, Optional, Union

# Third Party
import numpy as np
import statsmodels.api as sm
from caf.toolkit import BaseConfig
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class NorCom_inputs(BaseConfig):
    classified_build: Optional[Path] = None
    output_folder: Optional[Path] = None
    target_column: Optional[str] = None
    index_columns: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None
    weight_column: Optional[str] = None
    training_year: Optional[str] = None
    column_name_to_drop_rows: Optional[List[str]] = None
    value_in_row: Optional[List[Union[str, int, float]]] = None
    improve_data: Optional[str] = None
    model_type: Any
    skip_feature_selection: Optional[str] = None
    binary_prediction: Optional[str] = None


class Models(enum.Enum):
    LOGIT_REGRESSION_L1 = (LogisticRegression, {"penalty": "l1", "solver": "liblinear"})
    LOGIT_REGRESSION_L2 = (LogisticRegression, {"penalty": "l2"})
    LOGIT_REGRESSION_ELASTICNET = (
        LogisticRegression,
        {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
    )
    MULTINOMIAL = (LogisticRegression, {"multi_class": "multinomial", "solver": "lbfgs"})

    RANDOM_FOREST = RandomForestRegressor

    EXTRA_TREES = ExtraTreesRegressor

    GRADIENT_BOOSTING = GradientBoostingRegressor

    ADABOOST = AdaBoostRegressor

    BAGGING = BaggingRegressor

    SVR = SVR

    KNN = KNeighborsRegressor

    RIDGE = Ridge

    LASSO = Lasso

    ELASTICNET = ElasticNet

    LINEAR_REGRESSION = LinearRegression

    DECISION_TREE = DecisionTreeRegressor

    NEURAL_NETWORK = MLPRegressor

    PROBIT = sm.Probit

    RANDOM_FOREST_CLASSIFIER = RandomForestClassifier

    EXTRA_TREES_CLASSIFIER = ExtraTreesClassifier

    DECISION_TREE_CLASSIFIER = DecisionTreeClassifier

    def get_model_instance(self):
        model_class, params = self.value
        return model_class(**params) if model_class == LogisticRegression else model_class


Default_regression_methods = [Models.LASSO, Models.RIDGE, Models.ELASTICNET]
Models_List_ = [
    Models.RANDOM_FOREST,
    Models.EXTRA_TREES,
    Models.GRADIENT_BOOSTING,
    Models.ADABOOST,
    Models.BAGGING,
    Models.SVR,
    Models.KNN,
    Models.RIDGE,
    Models.LASSO,
    Models.ELASTICNET,
    Models.LINEAR_REGRESSION,
    Models.DECISION_TREE,
    Models.NEURAL_NETWORK,
    Models.LOGIT_REGRESSION_L1,
    Models.LOGIT_REGRESSION_L2,
    Models.LOGIT_REGRESSION_ELASTICNET,
    Models.PROBIT,
    Models.MULTINOMIAL,
    Models.RANDOM_FOREST_CLASSIFIER,
    Models.EXTRA_TREES_CLASSIFIER,
    Models.DECISION_TREE_CLASSIFIER,
]


class ModelGrids(enum.Enum):
    RANDOM_FOREST = {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    EXTRA_TREES = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    GRADIENT_BOOSTING = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
    }

    ADABOOST = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}

    BAGGING = {
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

    DECISION_TREE = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    LOGIT_REGRESSION_L1 = {"C": [1.0, 0.1, 0.01, 0.001]}
    LOGIT_REGRESSION_L2 = {"C": [1.0, 0.1, 0.01, 0.001]}
    LOGIT_REGRESSION_ELASTICNET = {"C": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}

    PROBIT = {"method": ["newton", "bfgs", "lbfgs"], "disp": [False]}

    # MULTINOMIAL = {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs", "newton-cg", "sag", "saga"]}
    MULTINOMIAL = {
        "C": np.arange(0.1, 10, 0.1).tolist(),
        "solver": ["lbfgs", "newton-cg", "sag", "saga"],
    }

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

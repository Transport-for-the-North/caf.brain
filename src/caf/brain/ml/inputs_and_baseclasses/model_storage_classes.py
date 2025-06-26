# -*- coding: utf-8 -*-
"""
Created on: 12/12/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import enum
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


class Models_storage(enum.Enum):
    """
    Model storage for caf.ml.
    """

    # # # REGRESSION # # #
    LOGIT_REGRESSION_LASSO = (LogisticRegression, {"penalty": "l1", "solver": "liblinear"})
    LOGIT_REGRESSION_RIDGE = (LogisticRegression, {"penalty": "l2"})
    LOGIT_REGRESSION_ELASTICNET = (
        LogisticRegression,
        {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
    )
    MULTINOMIAL = (LogisticRegression, {"multi_class": "multinomial", "solver": "lbfgs"})
    LINEAR_REGRESSION = LinearRegression

    # # # TREE REGRESSION # # #
    RANDOM_FOREST = RandomForestRegressor
    EXTRA_TREES = ExtraTreesRegressor
    DECISION_TREE = DecisionTreeRegressor

    # # # TREE REGRESSION CLASSIFIER # # #
    RANDOM_FOREST_CLASSIFIER = RandomForestClassifier
    EXTRA_TREES_CLASSIFIER = ExtraTreesClassifier
    DECISION_TREE_CLASSIFIER = DecisionTreeClassifier

    # # # ML SPECIFIC ALGORITHMS # # #
    GRADIENT_BOOSTING = GradientBoostingRegressor
    ADABOOST = AdaBoostRegressor
    BAGGING = BaggingRegressor
    SVR = SVR
    KNN = KNeighborsRegressor

    def get_model_storage(self):
        """
        Helper method for creating an instance of a model selected by the user.
        Returns:
            Model instance.
        """
        if isinstance(self.value, tuple):
            model_class, params = self.value
            return model_class(**params)
        else:
            return self.value()


class Models_grid_storage(enum.Enum):
    """
    Hyperparameter storage for caf.ml.
    """

    # # # REGRESSION # # #
    LOGIT_REGRESSION_LASSO = {"C": [1.0, 0.1, 0.01, 0.001]}
    LOGIT_REGRESSION_RIDGE = {"C": [1.0, 0.1, 0.01, 0.001]}
    LOGIT_REGRESSION_ELASTICNET = {"C": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}
    MULTINOMIAL = {
        "C": np.arange(0.1, 10, 0.1).tolist(),
        "solver": ["lbfgs", "newton-cg", "sag", "saga"],
    }

    # # # TREE REGRESSION # # #
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
    DECISION_TREE = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # # # TREE REGRESSION CLASSIFIER # # #
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

    # # # ML SPECIFIC ALGORITHMS # # #
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
    SVR = {"C": [0.1, 1, 10], "kernel": ["linear", "poly", "rbf"], "epsilon": [0.1, 0.2, 0.3]}
    KNN = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"], "p": [1, 2]}

    def get_model_grid(self):
        """
        Helper method to get a hyperparameter grid for the selected model.

        Returns:
            tuple: hyperparameters.
        """
        return self.value

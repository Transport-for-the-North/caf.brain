# -*- coding: utf-8 -*-
"""
input classes for caf.ml models
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position

import abc
from pathlib import Path
from caf.toolkit import BaseConfig
from typing import Optional, List, Any
import enum
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (KFold,
                                     RandomizedSearchCV,
                                     GridSearchCV,
                                     StratifiedKFold,
                                     RepeatedKFold,
                                     RepeatedStratifiedKFold)


class CarAccessInputs(BaseConfig):
    #### PROCESS DATA ####
    # imports
    x_path: Optional[Path] = None
    y_path: Optional[Path] = None
    folder_path: Optional[Path] = None

    # data sorting imports
    index_columns: Optional[List[str]] = None
    drop_columns: Optional[List[str]] = None
    keep_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    output_folder: Optional[Path] = None

    # wide to long imports
    wide_format: Optional[str] = None
    variable_name: Optional[str] = None
    value_name: Optional[str] = None

    # optional imports
    outlier_threshold: Optional[str] = None


    #### FEATURE SELECTION ####
    process_data_used: Optional[bool] = True




    simple_feature_selection: Optional[str] = None
    feature_selection_exhaustive: Optional[str] = None

    model_type: Any  # abc.ABCMeta not supported by caf.toolkit currently
    cv_method: Optional[str] = None
    splits: Optional[str] = None
    repeats: Optional[str] = None


    #### CROSS VALIDATION ####



    folder: Optional[Path] = None

    hp_optimisation: Optional[str] = None

    #### PREDICTION FUNCTIONS ####
    single_year_prediction: Optional[str] = None
    predict_data: Optional[Path] = None
    year_range: Optional[tuple[str]] = None
    index_columns_predict: Optional[List[str]] = None
    drop_columns_predict: Optional[List[str]] = None
    keep_columns_predict: Optional[List[str]] = None
    outlier_threshold_predict: Optional[str] = None

    #### STORE MODEL ####
    saved_model: Optional[str] = None


    threshold: Optional[float] = None
    threshold_corr: Optional[float] = None


class Models(enum.Enum):
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


Default_regression_methods = [Models.LASSO, Models.RIDGE, Models.ELASTICNET]

#todo include more simple models from biogem


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



class CV_models(enum.Enum):
    Kfold = KFold

    Stratified_kfold = StratifiedKFold

    Repeated_kfold = RepeatedKFold

    Repeated_stratified_kfold = RepeatedStratifiedKFold


class DefaultRegressionMethods(enum.Enum):
    LASSO = {
        "alpha": [0.1, 1.0, 10.0]
    }

    RIDGE = {
        "alpha": [0.1, 1.0, 10.0]
    }

    ELASTICNET = {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.5, 0.9]
    }

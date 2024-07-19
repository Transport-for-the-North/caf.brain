# -*- coding: utf-8 -*-
"""
input classes for caf_ml models
"""
import numpy as np
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm



import abc
from pathlib import Path
from caf.toolkit import BaseConfig
from typing import Optional, List, Any, Union
import enum
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor, RandomForestClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (KFold,
                                     RandomizedSearchCV,
                                     GridSearchCV,
                                     StratifiedKFold,
                                     RepeatedKFold,
                                     RepeatedStratifiedKFold, TimeSeriesSplit)


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

    # wide to long importshow can
    wide_format: Optional[str] = None
    variable_name: Optional[str] = None
    value_name: Optional[str] = None

    # optional imports
    outlier_threshold: Optional[int] = None


    #### FEATURE SELECTION ####
    #process_data_used: Optional[bool] = True
    model_type: Any  # abc.ABCMeta not supported by caf.toolkit currently
    cv_method: Optional[str] = None
    splits: Optional[str] = None
    repeats: Optional[str] = None


    #### PREDICTION FUNCTIONS ####
    single_year_prediction: Optional[str] = None
    predict_data: Optional[Path] = None
    year_range: Optional[tuple[str]] = None
    index_columns_predict: Optional[List[str]] = None
    drop_columns_predict: Optional[List[str]] = None
    keep_columns_predict: Optional[List[str]] = None
    outlier_threshold_predict: Optional[str] = None

    multiple_year_prediction: Optional[str] = None

    #### STORE MODEL ####
    saved_model: Optional[str] = None


    threshold: Optional[float] = None
    threshold_corr: Optional[float] = None

    validation_data: Optional[Path] = None
    skip_feature_selection: Optional[str] = None
    skip_data_analysis: Optional[str] = None
    skip_hyperparameter_optimisation: Optional[str] = None
    basic_model: Optional[str] = None

    categorical_data: Optional[str] = None
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    categorical_target: Optional[str] = None

    column_name_to_drop_rows: Optional[List[str]] = None
    value_in_row: Optional[List[Union[str, int, float]]] = None


class Models(enum.Enum):
    LOGIT_REGRESSION_L1 = (LogisticRegression, {'penalty': 'l1', 'solver': 'liblinear'})
    LOGIT_REGRESSION_L2 = (LogisticRegression, {'penalty': 'l2'})
    LOGIT_REGRESSION_ELASTICNET = (LogisticRegression, {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5})
    MULTINOMIAL = (LogisticRegression, {'multi_class': 'multinomial', 'solver': 'lbfgs'})

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
Models_List_ = [Models.RANDOM_FOREST, Models.EXTRA_TREES, Models.GRADIENT_BOOSTING, Models.ADABOOST,
               Models.BAGGING, Models.SVR, Models.KNN, Models.RIDGE, Models.LASSO, Models.ELASTICNET,
               Models.LINEAR_REGRESSION, Models.DECISION_TREE, Models.NEURAL_NETWORK, Models.LOGIT_REGRESSION_L1,
                Models.LOGIT_REGRESSION_L2, Models.LOGIT_REGRESSION_ELASTICNET, Models.PROBIT, Models.MULTINOMIAL, Models.RANDOM_FOREST_CLASSIFIER,
                Models.EXTRA_TREES_CLASSIFIER, Models.DECISION_TREE_CLASSIFIER]


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

    LOGIT_REGRESSION_L1 = {"C": [0.1, 1.0, 10.0]}
    LOGIT_REGRESSION_L2 = {"C": [0.1, 1.0, 10.0]}
    LOGIT_REGRESSION_ELASTICNET = {"C": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}

    PROBIT = {"method": ["newton", "bfgs", "lbfgs"], "disp": [False]}

    #MULTINOMIAL = {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs", "newton-cg", "sag", "saga"]}
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


class CV_models(enum.Enum):
    Kfold = KFold

    Stratified_kfold = StratifiedKFold

    Repeated_kfold = RepeatedKFold

    Repeated_stratified_kfold = RepeatedStratifiedKFold

    Time_series_split = TimeSeriesSplit


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

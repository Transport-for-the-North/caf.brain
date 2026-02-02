"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""

# Built-Ins
import enum
from pathlib import Path
from typing import List, Optional, Type, Union

# Third Party
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from caf.toolkit import BaseConfig


class Models(enum.Enum):
    """
    Available algorithms for the caf.brAIn prediction model.
    """

    LOGIT_REGRESSION_L1 = "logit_regression_l1"
    LOGIT_REGRESSION_L2 = "logit_regression_l2"
    LOGIT_REGRESSION_ELASTICNET = "logit_regression_elasticnet"
    MULTINOMIAL = "multinomial"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    EXTRA_TREES_REGRESSOR = "extra_trees_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    ADABOOST_REGRESSOR = "adabost_regressor"
    BAGGING_REGRESSOR = "bagging_regressor"
    SVR = "svr"
    KNN = "knn"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTICNET = "elasticnet"
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    EXTRA_TREES_CLASSIFIER = "extra_trees_classifier"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    SVM_CLASSIFIER = "svm_classifier"

    def get_model(self):
        """
        Helper method for algorithm selection and initialisation.

        Returns
        -------
        Tuple of initialised model and empty hyperparameter dictionary.
        """
        model_class, params = MODEL_LOOKUP[self]
        return model_class(**params)


class Paths(BaseConfig):
    """
    Prediction model inputs that are paths to external files.
    """

    file_path: Optional[Path] = None
    folder_path: Optional[Path] = None
    output_path: Optional[Path] = None
    validation_path: Optional[Path] = None


class DataClassificationInputs(BaseConfig):
    """
    Inputs that help define and outline the structure of the input data.
    """

    target_column: Optional[str] = None
    custom_index: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None
    weight_column: Optional[str] = None
    is_time_series: Optional[bool] = False


class TransformingInputDataInputs(BaseConfig):
    """
    Inputs that dictate how the data is transformed for machine learning
    modelling.
    """

    column_name_to_drop_rows: Optional[List[str]] = None
    value_in_row: Optional[List[Union[str, int, float]]] = None
    classification_prediction: Optional[tuple[int, ...]] = None
    split_by_value: Optional[str] = None
    split_size: Optional[float] = None
    sample_size_encode: Optional[bool] = False
    select_encode_values: Optional[bool] = False
    encode_values_to_drop: Optional[List[str]] = None


class ModellingInputs(BaseConfig):
    """
    Inputs that control the machine learning modelling pipeline and _functions.
    """

    model_choice: Optional[List[Models]] = None
    full_transformations: Optional[bool] = False
    cv: Optional[str] = None
    skip_feature_selection: Optional[bool] = False
    intensive_feature_selection: Optional[bool] = False


class PredictionModelInputs(BaseConfig):
    """
    Caf.brAIn prediction model inputs.
    """

    paths: Paths = Paths()
    data_classification: DataClassificationInputs = DataClassificationInputs()
    transforming_inputs: TransformingInputDataInputs = TransformingInputDataInputs()
    modelling: ModellingInputs = ModellingInputs()


MODEL_LOOKUP = {
    Models.LOGIT_REGRESSION_L1: (
        LogisticRegression,
        {"penalty": "l1", "solver": "liblinear"},
    ),
    Models.LOGIT_REGRESSION_L2: (
        LogisticRegression,
        {"penalty": "l2"},
    ),
    Models.LOGIT_REGRESSION_ELASTICNET: (
        LogisticRegression,
        {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
    ),
    Models.MULTINOMIAL: (
        LogisticRegression,
        {"multi_class": "multinomial", "solver": "lbfgs"},
    ),
    Models.RANDOM_FOREST_REGRESSOR: (
        RandomForestRegressor,
        {},
    ),
    Models.EXTRA_TREES_REGRESSOR: (
        ExtraTreesRegressor,
        {},
    ),
    Models.GRADIENT_BOOSTING_REGRESSOR: (
        GradientBoostingRegressor,
        {},
    ),
    Models.GRADIENT_BOOSTING_CLASSIFIER: (
        GradientBoostingClassifier,
        {},
    ),
    Models.ADABOOST_REGRESSOR: (
        AdaBoostRegressor,
        {},
    ),
    Models.BAGGING_REGRESSOR: (BaggingRegressor, {}),
    Models.SVR: (SVR, {}),
    Models.KNN: (KNeighborsRegressor, {}),
    Models.RIDGE: (Ridge, {}),
    Models.LASSO: (Lasso, {}),
    Models.ELASTICNET: (ElasticNet, {}),
    Models.LINEAR_REGRESSION: (LinearRegression, {}),
    Models.DECISION_TREE_REGRESSOR: (
        DecisionTreeRegressor,
        {},
    ),
    Models.RANDOM_FOREST_CLASSIFIER: (
        RandomForestClassifier,
        {},
    ),
    Models.EXTRA_TREES_CLASSIFIER: (
        ExtraTreesClassifier,
        {},
    ),
    Models.DECISION_TREE_CLASSIFIER: (
        DecisionTreeClassifier,
        {},
    ),
    Models.SVM_CLASSIFIER: (
        OneVsRestClassifier,
        {"estimator": LinearSVC()},
    ),
}


class ModelGrids(enum.Enum):
    """
    Hyperparameter grids for available models from the Models
    enum class.
    """

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

    GRADIENT_BOOSTING_CLASSIFIER = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7],
    }

    SVM_CLASSIFIER = {
        "estimator__C": [0.1, 1, 10],
        "estimator__loss": ["hinge", "squared_hinge"],
    }

    @classmethod
    def get_grid(cls, model_enum):
        return getattr(cls, model_enum.name).value


def get_model_grid_from_type(model_type: Type[BaseEstimator]) -> dict:
    """
    Retrieve the hyperparameter grid for a given scikit-learn model class.

    This function maps a model class (e.g., RandomForestRegressor) to its
    predefined hyperparameter grid stored in the `ModelGrids` enum. It is used
    to support automated hyperparameter tuning workflows.

    Parameters
    ----------
    model_type: The scikit-learn model class for which the hyperparameter grid
                is requested. This should be a class (not an instance).

    Returns
    -------
    A dictionary containing the hyperparameter grid for the specified model class.
    """

    model_type_to_parm_grid = {
        RandomForestRegressor: ModelGrids.RANDOM_FOREST_REGRESSOR.value,
        ExtraTreesRegressor: ModelGrids.EXTRA_TREES_REGRESSOR.value,
        GradientBoostingRegressor: ModelGrids.GRADIENT_BOOSTING_REGRESSOR.value,
        AdaBoostRegressor: ModelGrids.ADABOOST_REGRESSOR.value,
        BaggingRegressor: ModelGrids.BAGGING_REGRESSOR.value,
        SVR: ModelGrids.SVR.value,
        KNeighborsRegressor: ModelGrids.KNN.value,
        Ridge: ModelGrids.RIDGE.value,
        Lasso: ModelGrids.LASSO.value,
        ElasticNet: ModelGrids.ELASTICNET.value,
        DecisionTreeRegressor: ModelGrids.DECISION_TREE_REGRESSOR.value,
        RandomForestClassifier: ModelGrids.RANDOM_FOREST_CLASSIFIER.value,
        ExtraTreesClassifier: ModelGrids.EXTRA_TREES_CLASSIFIER.value,
        DecisionTreeClassifier: ModelGrids.DECISION_TREE_CLASSIFIER.value,
        GradientBoostingClassifier: ModelGrids.GRADIENT_BOOSTING_CLASSIFIER.value,
        LogisticRegression: ModelGrids.LOGIT_REGRESSION_ELASTICNET.value,
    }
    try:
        param_grid = model_type_to_parm_grid[model_type]
        assert isinstance(param_grid, dict)
        return param_grid
    except KeyError as e:
        raise ValueError(f"No grid found for model instance of type {model_type}") from e

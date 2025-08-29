"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""

# Built-Ins
import enum
from pathlib import Path
from typing import Any, List, Optional, Type, Union

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


class PredictionModelInputs(BaseConfig):
    """
    Caf.brAIn prediction model inputs.
    """

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
        Inputs that control the machine learning modelling pipeline and functions.
        """

        model_choice: Optional[List[Any]] = None
        full_transformations: Optional[bool] = False
        cv: Optional[str] = None
        skip_feature_selection: Optional[bool] = False
        intensive_feature_selection: Optional[bool] = False

    paths: Paths = Paths()
    data_classification: DataClassificationInputs = DataClassificationInputs()
    transforming_inputs: TransformingInputDataInputs = TransformingInputDataInputs()
    modelling: ModellingInputs = ModellingInputs()


class Models(enum.Enum):
    """
    Available algorithms for the caf.brAIn prediction model.
    """

    LOGIT_REGRESSION_L1 = (
        LogisticRegression,
        {"penalty": "l1", "solver": "liblinear"},
    )  # type: ignore
    LOGIT_REGRESSION_L2 = (
        LogisticRegression,
        {"penalty": "l2"},
    )  # type: ignore
    LOGIT_REGRESSION_ELASTICNET = (
        LogisticRegression,
        {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
    )  # type: ignore
    MULTINOMIAL = (
        LogisticRegression,
        {"multi_class": "multinomial", "solver": "lbfgs"},
    )  # type: ignore
    RANDOM_FOREST_REGRESSOR = (
        RandomForestRegressor,
        {},
    )  # type: ignore
    EXTRA_TREES_REGRESSOR = (
        ExtraTreesRegressor,
        {},
    )  # type: ignore
    GRADIENT_BOOSTING_REGRESSOR = (
        GradientBoostingRegressor,
        {},
    )  # type: ignore
    GRADIENT_BOOSTING_CLASS = (
        GradientBoostingClassifier,
        {},
    )  # type: ignore
    ADABOOST_REGRESSOR = (
        AdaBoostRegressor,
        {},
    )  # type: ignore
    BAGGING_REGRESSOR = (BaggingRegressor, {})  # type: ignore
    SVR = (SVR, {})  # type: ignore
    KNN = (KNeighborsRegressor, {})  # type: ignore
    RIDGE = (Ridge, {})  # type: ignore
    LASSO = (Lasso, {})  # type: ignore
    ELASTICNET = (ElasticNet, {})  # type: ignore
    LINEAR_REGRESSION = (LinearRegression, {})  # type: ignore
    DECISION_TREE_REGRESSOR = (
        DecisionTreeRegressor,
        {},
    )  # type: ignore
    RANDOM_FOREST_CLASSIFIER = (
        RandomForestClassifier,
        {},
    )  # type: ignore
    EXTRA_TREES_CLASSIFIER = (
        ExtraTreesClassifier,
        {},
    )  # type: ignore
    DECISION_TREE_CLASSIFIER = (
        DecisionTreeClassifier,
        {},
    )  # type: ignore
    SVM_CLASSIFIER = (
        OneVsRestClassifier,
        {"estimator": LinearSVC()},
    )  # type: ignore
    # STATS_OLS_REGRESSOR = (sm.OLS, {})  # type: ignore
    # STATS_MLR_REGRESSOR = (sm.RLM, {})  # type: ignore
    # STATS_LOGISTIC_CLASSIFIER = (sm.Logit, {})  # type: ignore
    # STATS_PROBIT_CLASSIFIER = (sm.Probit, {})  # type: ignore
    # STATS_POISSON_REGRESSOR = (
    #     sm.GLM,
    #     {"family": sm.families.Poisson()},
    # )  # type: ignore
    # STATS_NEGATIVE_BINOMIAL_REGRESSOR = (
    #     sm.GLM,
    #     {"family": sm.families.NegativeBinomial()},
    # )  # type: ignore
    # STATS_LINEAR_EFFECTS_REGRESSOR = (sm.MixedLM, {})  # type: ignore
    # STATS_ARIMA_REGRESSOR = (sm.tsa.ARIMA, {})  # type: ignore
    # STATS_SARIMA_REGRESSOR = (sm.tsa.SARIMAX, {})  # type: ignore
    # STATS_MULTINOMIAL_LOGISTIC_CLASSIFIER = (
    #     sm.MNLogit,
    #     {},
    # )  # type: ignore
    # STATS_ORDINAL_LOGISTIC_CLASSIFIER = (
    #     OrderedModel,
    #     {"distr": "logit"},
    # )  # type: ignore
    # STATS_TOBIT_REGRESSOR = (sm.Tobit, {})?

    def get_model(self):
        """
        Helper method for algorithm selection and initialisation.

        Returns
        -------
        Tuple of initialised model and empty hyperparameter dictionary.
        """
        model_class, params = self.value if isinstance(self.value, tuple) else (self.value, {})
        return model_class(**params)


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

    # # Statsmodels below:
    # STATS_OLS_REGRESSOR = {
    #     'missing': ['none', 'drop', 'raise'],
    #     'hasconst': [True, False]
    # }
    #
    # STATS_MLR_REGRESSOR = {
    #     'M': ['huber', 'bisquare', 'andrew'],
    #     'tune': [1.345, 2.0, 4.0],
    #     'maxiter': [50, 100, 200]
    # }
    #
    # STATS_LOGISTIC_CLASSIFIER = {
    #     'method': ['newton', 'bfgs', 'lbfgs', 'powell'],
    #     'maxiter': [100, 200, 500],
    #     'tol': [1e-4, 1e-5, 1e-6]
    # }
    #
    # STATS_PROBIT_CLASSIFIER = {
    #     'method': ['newton', 'bfgs', 'powell'],
    #     'maxiter': [100, 200, 500],
    #     'tol': [1e-4, 1e-5, 1e-6]
    # }
    #
    # STATS_POISSON_REGRESSOR = {
    #     'alpha': [0, 0.1, 1.0],
    #     'L1_wt': [0, 0.5, 1.0],
    #     'maxiter': [100, 200, 500]
    # }
    #
    # STATS_NEGATIVE_BINOMIAL_REGRESSOR = {
    #     'alpha': [0, 0.1, 1.0],
    #     'L1_wt': [0, 0.5, 1.0],
    #     'maxiter': [100, 200, 500]
    # }
    #
    # STATS_LINEAR_EFFECTS_REGRESSOR = {
    #     'method': ['lbfgs', 'cg'],
    #     'maxiter': [100, 200, 500],
    #     'reml': [True, False]
    # }
    #
    # STATS_ARIMA_REGRESSOR = {
    #     'order_p': [0, 1, 2],
    #     'order_d': [0, 1],
    #     'order_q': [0, 1, 2],
    #     'method': ['css-mle', 'mle', 'css']
    # }
    #
    # STATS_SARIMA_REGRESSOR = {
    #     'order_p': [0, 1, 2],
    #     'order_d': [0, 1],
    #     'order_q': [0, 1, 2],
    #     'seasonal_order_P': [0, 1],
    #     'seasonal_order_D': [0, 1],
    #     'seasonal_order_Q': [0, 1],
    #     'seasonal_periods': [4, 12]
    # }
    #
    # STATS_MULTINOMIAL_LOGISTIC_CLASSIFIER = {
    #     'method': ['newton', 'bfgs', 'lbfgs'],
    #     'maxiter': [100, 200, 500],
    #     'tol': [1e-4, 1e-5, 1e-6]
    # }
    #
    # STATS_ORDINAL_LOGISTIC_CLASSIFIER = {
    #     'method': ['bfgs', 'newton'],
    #     'maxiter': [100, 200, 500],
    #     'distr': ['logit', 'probit']
    # }
    #
    # STATS_TOBIT_REGRESSOR = {
    #     'method': ['powell', 'bfgs', 'newton'],
    #     'maxiter': [100, 200, 500],
    #     'tol': [1e-4, 1e-5, 1e-6]
    # }

    @classmethod
    def get_grid(cls, model_enum):
        """
        Parameters
        ----------
        model_enum:

        Returns
        -------

        """
        return getattr(cls, model_enum.name).value


def get_model_grid_from_type(model_type: Type[BaseEstimator]) -> dict:
    """
    Retrieve the hyperparameter grid for a given scikit-learn model class.

    This function maps a model class (e.g., `RandomForestRegressor`) to its predefined
    hyperparameter grid stored in the `ModelGrids` enum. It is used to support automated
    hyperparameter tuning workflows such as grid search.

    Parameters
    ----------
    model_type : Type[BaseEstimator]
        The scikit-learn model class for which the hyperparameter grid is requested.
        This should be a class (not an instance) that inherits from `BaseEstimator`.

    Returns
    -------
    dict:
        A dictionary containing the hyperparameter grid for the specified model class.

    Raises
    ------
    ValueError
        If the model type is not found in the grid mapping.
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
        # KLUDGE: Need a better way to handle the different logistic regressions
        LogisticRegression: ModelGrids.LOGIT_REGRESSION_ELASTICNET.value,
    }
    try:
        param_grid = model_type_to_parm_grid[model_type]
        assert isinstance(param_grid, dict)
        return param_grid
    except KeyError as e:
        raise ValueError(f"No grid found for model instance of type {model_type}") from e

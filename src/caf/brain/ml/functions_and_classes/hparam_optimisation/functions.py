"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""

import logging
import gc
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from caf.brain.ml.inputs_and_baseclasses.ml_inputs import (
    ModelGrids,
    get_model_grid,
)
from caf.brain.ml.functions_and_classes.feature_selection.functions import (
    get_cv_class,
)

LOG = logging.getLogger(__name__)


def select_param(
    train_final: pd.DataFrame,
    target_column: str,
    model_instance,
    model_name,
    classification_prediction: tuple[int, ...],
    cv: str,
    weight_column: str,
    output_folder: Path,
    is_time_series: bool,
):
    """
    Hyperparameter optimisation based on the ModelGrids Enum class.

    Parameters
    ----------
    train_final: Dataframe of final training data post feature selection.
    target_column: String column name of value to predict.
    model_instance: Initialised model algorithm from Models enum class.
    model_name: List or one algorithm to use as the base of the model.
                Available algorithms can be seen in ml_inputs.py.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    cv: Cross validation method passed as a string. Any popular
        SciKitlearn methods are suitable with KFold being default if
        left as None.
    weight_column: Optional string column value to be used as weight.
    output_folder: Path to output location.
    is_time_series: If true then data must be time series. Time series
                    based characteristics are taken into consideration
                    during function execution.

    Returns
    -------
    best_model: Fitted final model for prediction on unseen (test) data.

    """
    x = train_final.drop(columns=[target_column] + ([weight_column] if weight_column else []))
    y = train_final[target_column]
    weight = train_final[weight_column].values.flatten() if weight_column else None
    cv = get_cv_class(cv_method=cv, splits=None, repeats=None, is_time_series=is_time_series)

    if len(model_name) == 1:
        param_grid = ModelGrids.get_grid(model_name[0])
    else:
        param_grid = get_model_grid(model_instance)

    if isinstance(model_instance, LinearRegression):
        LOG.info(
            "No hyperparameters in Linear Regression. Skipping hyperparameter optimisation."
        )
        return model_instance

    if classification_prediction is not None:
        if isinstance(
            model_instance,
            (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier),
        ):
            if isinstance(model_instance, DecisionTreeClassifier):
                model_instance.set_params(max_depth=10)
                best_params = rand_search(
                    model_instance=model_instance,
                    param_grid=param_grid,
                    cv=cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    weight=weight,
                    x=x,
                    y=y,
                )
            else:
                model_instance.set_params(n_estimators=10, n_jobs=-1)
                best_params = perform_grid_search(
                    model_instance=model_instance,
                    param_grid=param_grid,
                    cv=cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    weight=weight,
                    x=x,
                    y=y,
                )
        else:
            best_params = perform_grid_search(
                model_instance=model_instance,
                param_grid=param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                weight=weight,
                x=x,
                y=y,
            )

    else:
        if isinstance(
            model_instance, (RandomForestRegressor, ExtraTreesRegressor, DecisionTreeRegressor)
        ):
            if isinstance(model_instance, DecisionTreeRegressor):
                model_instance.set_params(max_depth=10)
                best_params = rand_search(
                    model_instance=model_instance,
                    param_grid=param_grid,
                    cv=cv,
                    scoring="r2",
                    n_jobs=-1,
                    weight=weight,
                    x=x,
                    y=y,
                )
            else:
                model_instance.set_params(n_estimators=10, n_jobs=-1)
                best_params = perform_grid_search(
                    model_instance=model_instance,
                    param_grid=param_grid,
                    cv=cv,
                    scoring="r2",
                    n_jobs=-1,
                    weight=weight,
                    x=x,
                    y=y,
                )
        else:
            best_params = perform_grid_search(
                model_instance=model_instance,
                param_grid=param_grid,
                cv=cv,
                scoring="r2",
                n_jobs=-1,
                weight=weight,
                x=x,
                y=y,
            )

    best_model = model_instance.set_params(**best_params)
    best_model.fit(x, y, sample_weight=weight)

    model_filename = os.path.join(output_folder, "final_model.pkl")
    joblib.dump(best_model, model_filename)
    LOG.info("Best model saved here: %s", model_filename)
    # coeffs
    if hasattr(best_model, "coef_"):
        coefficients = best_model.coef_
        coefficients = np.squeeze(coefficients)

        if coefficients.ndim == 1:
            coeff_df = pd.DataFrame({"Feature": x.columns, "Coefficient": coefficients})
        else:
            coeff_df = pd.DataFrame(coefficients.T, columns=x.columns)
            coeff_df.insert(0, "Feature", x.columns)

        coeff_df.to_csv(
            os.path.join(output_folder, "final_model_coefficients.csv"), index=False
        )

    return best_model


def rand_search(
    model_instance,
    param_grid,
    cv,
    scoring: str,
    n_jobs: int,
    weight: pd.Series,
    x: pd.DataFrame,
    y: pd.DataFrame,
):
    """
    Helper function to conduct randomised search of hyperparameters.

    Parameters
    ----------
    model_instance: Initialised model algorithm from Models enum class.
    param_grid: Instance of Enum class ModelGrids. Grid of hyperparameters.
    cv: Cross validation method passed as a string. Any popular
        SciKitlearn methods are suitable with KFold being default if
        left as None.
    scoring: Scoring method used when evaluating hyperparameters.
    n_jobs: Integer to represent number of cores used to evaluate hyperparameters.
    weight: Weight values in series form.
    x: Training data split into explanatory variables only.
    y: Training data split only into the target variable.

    Returns
    -------
    best_params: best hyperparameters found.

    """
    random_search = RandomizedSearchCV(
        model_instance,
        param_grid,
        cv=cv,
        scoring=scoring,
        verbose=2,
        n_jobs=n_jobs,
        n_iter=10,
        return_train_score=False,
        pre_dispatch="1*n_jobs",
    )
    gc.collect()
    random_search.fit(x, y, sample_weight=weight)
    best_params = random_search.best_params_
    LOG.info("Best parameters for model are: %s", best_params)
    LOG.info("CV results: %s", random_search.cv_results_)
    return best_params


def perform_grid_search(
    model_instance,
    param_grid,
    cv,
    scoring: str,
    n_jobs: int,
    weight: pd.Series,
    x: pd.DataFrame,
    y: pd.DataFrame,
):
    """
    Helper function to conduct grid search of hyperparameters.

    Parameters
    ----------
    model_instance: Initialised model algorithm from Models enum class.
    param_grid: Instance of Enum class ModelGrids. Grid of hyperparameters
    cv: Cross validation method passed as a string. Any popular
        SciKitlearn methods are suitable with KFold being default if
        left as None.
    scoring: Scoring method used when evaluating hyperparameters.
    n_jobs: Integer to represent number of cores used to evaluate hyperparameters.
    weight: Weight values in series form.
    x: Training data split into explanatory variables only.
    y: Training data split only into the target variable.

    Returns
    -------
    best_params: best hyperparameters found.
    """
    grid_search = GridSearchCV(
        model_instance,
        param_grid,
        cv=cv,
        scoring=scoring,
        verbose=2,
        n_jobs=n_jobs,
        return_train_score=False,
        pre_dispatch="1*n_jobs",
    )
    gc.collect()
    grid_search.fit(x, y, sample_weight=weight)
    best_params = grid_search.best_params_
    LOG.info("Best parameters for model are: %s", best_params)
    LOG.info("CV results: %s", grid_search.cv_results_)
    return best_params

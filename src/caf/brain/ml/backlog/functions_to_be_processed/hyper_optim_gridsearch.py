# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
"""
Created on: 1/31/2024
Original author: Adil Zaheer
"""
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    ParameterGrid,
    TimeSeriesSplit,
    KFold,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from caf.ml.backlog.old_inputs import Models, ModelGrids
import statsmodels.api as sm


def get_model_name(model_instance):
    model_class = type(model_instance)
    instance_params = model_instance.get_params()

    for name, member in Models.__members__.items():
        if isinstance(member.value, tuple):
            expected_class, expected_params = member.value
            if expected_class == model_class:
                default_params = expected_class().get_params()
                combined_params = {**default_params, **expected_params}

                print(f"Checking {name} with combined parameters: {combined_params}")

                # Matching all combined parameters
                if all(
                    instance_params.get(key) == value for key, value in combined_params.items()
                ):
                    return name
        else:
            if member.value == model_class:
                return name
    return None


def select_param(data, target_column, model, categorical_data, multiple_year_prediction):

    x = data.drop(columns=[target_column])
    y = data[target_column]

    print(f"Model instance: {model}")

    model_name = get_model_name(model_instance=model)
    print(f"Model name: {model_name}")

    cv = (
        TimeSeriesSplit(n_splits=5)
        if multiple_year_prediction is not None
        else KFold(n_splits=5)
    )

    # model_instance = Models[model_name].value()
    model_value = Models[model_name].value
    if isinstance(model_value, tuple):
        model_class, params = model_value
        model_instance = model_class(**params)
    else:
        model_instance = model_value()

    if model_name in ModelGrids.__members__:
        param_grid = ModelGrids[model_name].value
    else:
        raise ValueError("Invalid regression method or hyperparameters not defined.")

    if isinstance(model_instance, LinearRegression):
        best_params = None
        print("No hyperparameters in Linear Regression. Skipping hyperparameter optimisation.")
        return best_params

    def perform_grid_search(model_instance, param_grid, cv, scoring):
        grid_search = GridSearchCV(
            model_instance, param_grid, cv=cv, scoring=scoring, verbose=2
        )
        grid_search.fit(x, y)
        best_params = grid_search.best_params_
        print("Best parameters for model are:")
        print(best_params)
        print("CV results:")
        print(grid_search.cv_results_)
        return best_params

    def perform_random_search(model_instance, param_grid, cv, scoring):
        random_search = RandomizedSearchCV(
            model_instance, param_distributions=param_grid, cv=cv, scoring=scoring, verbose=2
        )
        random_search.fit(x, y)
        best_params = random_search.best_params_
        print("Best parameters for model are:")
        print(best_params)
        print("CV results:")
        print(random_search.cv_results_)
        print("---------------------------------")
        print("hyperparameter optimisation is finished")
        print("---------------------------------")
        return best_params

    if categorical_data is not None:
        if model_name in [
            "LOGIT_REGRESSION_L1",
            "LOGIT_REGRESSION_L2",
            "LOGIT_REGRESSION_ELASTICNET",
            "MULTINOMIAL",
        ]:
            print("---------------------------------")
            print("Categorical hyperparameter optimisation starting")
            print("---------------------------------")
            if model_name == "LOGIT_REGRESSION_L1":
                model_instance = LogisticRegression(penalty="l1", solver="saga")
            elif model_name == "LOGIT_REGRESSION_L2":
                model_instance = LogisticRegression(penalty="l2")
            elif model_name == "LOGIT_REGRESSION_ELASTICNET":
                model_instance = LogisticRegression(penalty="elasticnet", solver="saga")
            elif model_name == "MULTINOMIAL":
                model_instance = LogisticRegression(multi_class="multinomial", solver="lbfgs")
            param_grid = ModelGrids[model_name].value
            return perform_grid_search(model_instance, param_grid, cv=cv, scoring="accuracy")

        if model_name == "PROBIT":
            param_grid = ModelGrids[model_name].value
            best_score = -float("inf")
            best_params = None

            for params in ParameterGrid(param_grid):
                probit_model = sm.Probit(x, y)
                result = probit_model.fit(method=params["method"], disp=params["disp"])
                score = result.prsquared

                if score > best_score:
                    best_score = score
                    best_params = params

            print("Best parameters for Probit model are:")
            print(best_params)
            return best_params

        if isinstance(
            model_instance,
            (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier),
        ):
            if isinstance(model_instance, DecisionTreeClassifier):
                model_instance.set_params(max_depth=10)
                return perform_random_search(
                    model_instance, param_grid, cv=cv, scoring="accuracy"
                )
            else:
                model_instance.set_params(n_estimators=10, n_jobs=-1)
                return perform_grid_search(
                    model_instance, param_grid, cv=cv, scoring="accuracy"
                )

    if categorical_data is None:
        if isinstance(
            model_instance, (RandomForestRegressor, ExtraTreesRegressor, DecisionTreeRegressor)
        ):
            if isinstance(model_instance, DecisionTreeRegressor):
                model_instance.set_params(max_depth=10)
                return perform_random_search(model_instance, param_grid, cv=cv, scoring="r2")

            else:
                model_instance.set_params(n_estimators=10, n_jobs=-1)
                return perform_random_search(model_instance, param_grid, cv=cv, scoring="r2")

        else:
            return perform_grid_search(model_instance, param_grid, cv=cv, scoring="r2")

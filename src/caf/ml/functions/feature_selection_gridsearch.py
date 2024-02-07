# -*- coding: utf-8 -*-
"""
Created on: 1/31/2024
Updated on:

Original author: Adil Zaheer
Last update made by:
Other updates made by: Isaac Scott

File purpose:

"""
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


def select_model(
    x: pd.DataFrame,
    y: pd.DataFrame,
    model_names: list[str] = [en.name for en in list(Models)],
):
    """
    This function scores different types of user specified models in order to
    find the best performing model for your data. The model comparison is
    based upon their mean r2 score from 5 fold cross validation. The
    standard deviation of the final model is also is shown.

    Parameters
    ----------

    x: The input features
    y: The target variable
    model_names: These are the user specified models that will be
    tested

    Returns
    -------
    The function returns the name of the best performing model
    """
    acc = {}
    score = 0
    # Set dummy variable to return if it fails.
    return_model = "dummy"
    for name in model_names:
        model = Models[name].value()
        scores = cross_val_score(model, x, y, cv=5, scoring="r2")
        mean = scores.mean()
        acc[name] = mean
        if mean > score:
            score = mean
            return_model = name
    return return_model


def select_param(x, y, model_name):
    """
    This function utilises gridsearch in order to find the best x
    parameters. The parameters are returned and are then used within the final
    forecasting model. See PARAM_GRIDS at the top of selection.py to edit
    how the functionality of the gridsearch.
    :param x: The input features
    :param y: The target variable
    :param model_name: Name of model that will be used to conduct the
    gridsearch. In the case of this model, extratreesregressor is used as
    through prior testing this provided the best r2 results.
    :return:
    """
    if model_name == "dummy":
        return None
    estimator = Models[model_name].value()
    grid = ModelGrids[model_name].value
    grid_search = GridSearchCV(estimator, grid, cv=5, scoring="r2")
    grid_search.fit(x, y)
    best_params = grid_search.best_params_
    return best_params

# -*- coding: utf-8 -*-
"""
Created on: 2/13/2024
Original author: Adil Zaheer
"""
import pandas as pd
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from caf.ml.inputs.cafml_inputs import CarAccessInputs, Models


def select_model(
    x: pd.DataFrame,
    y: pd.DataFrame,
    models_to_test: list[Models],
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
    models_to_test: These are the user specified models that will be
    tested

    Returns
    -------
    The function returns the name of the best performing model
    """

    acc = {}
    score = 0
    return_model = None
    for model_enum in models_to_test:
        model_instance = model_enum.value()
        model_name = model_enum.name.lower()
        scores = cross_val_score(model_instance, x, y, cv=5, scoring="r2")
        mean = scores.mean()
        acc[model_name] = mean
        if mean > score:
            score = mean
            return_model = model_instance # it was this here, change it: model_enum
    print(f"Best model score: {score}")

    return return_model

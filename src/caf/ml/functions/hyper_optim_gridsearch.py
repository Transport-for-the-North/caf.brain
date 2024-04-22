# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
"""
Created on: 1/31/2024
Original author: Adil Zaheer
"""
from sklearn.model_selection import GridSearchCV
from caf.ml.inputs.cafml_inputs import Models, ModelGrids


def get_model_name(model_instance):
    model_class = type(model_instance)
    for name, member in Models.__members__.items():
        if member.value == model_class:
            return name
    return None


def select_param(data, target_column, model):
    x = data.drop(columns=[target_column])
    y = data[target_column]

    model_name = get_model_name(model)

    if model_name in Models.__members__:
        model_instance = Models[model_name].value()
        param_grid = ModelGrids[model_name].value
    else:
        raise ValueError("Invalid regression method.")

    grid_search = GridSearchCV(model_instance, param_grid, cv=5, scoring="r2")
    grid_search.fit(x, y)

    best_params = grid_search.best_params_
    print('Best parameters for model are:')
    print(best_params)
    print('CV results:')
    print(grid_search.cv_results_)

    return best_params

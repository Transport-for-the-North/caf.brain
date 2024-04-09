# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
"""
Created on: 1/31/2024
Original author: Adil Zaheer
Other updates made by: Isaac Scott
"""
from sklearn.model_selection import GridSearchCV
from caf.ml.inputs.cafml_inputs import Models, ModelGrids, Default_regression_methods, DefaultRegressionMethods
from caf.ml.functions.model_algorithm_evaluation import select_model
from typing import Union, List


def select_param(data,
                 target_column: str,
                 regression_method: Union[Models, None],
                 model_type: Union[List[Models], None]
                 ):

    if data is not None:
        x = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        raise ValueError("Please provide a DataFrame.")

    if isinstance(regression_method, type):  # Check if regression_method is a class
        # Map regression class to corresponding enum member
        selected_model = next((m for m in Models if m.value == regression_method), None)
        if selected_model is None:
            raise ValueError("Invalid regression method.")
    else:
        raise ValueError("Invalid regression method type.")

    grid = ModelGrids[selected_model.name].value
    print('Selected model:', selected_model)
    print('Grid:', grid)

    estimator = selected_model
    grid_search = GridSearchCV(estimator(), grid, cv=5, scoring="r2")
    grid_search.fit(x, y)
    best_params = grid_search.best_params_
    print('Best parameters for model are:')
    print(best_params)
    print('CV results:')
    print(grid_search.cv_results_)
    return best_params

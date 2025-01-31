# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import gc
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor,
                              ExtraTreesRegressor,
                              ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from caf.ml.MODELS.prediction_model.prediction_model_inputs import ModelGrids, get_model_grid
from caf.ml.feature_selection.feature_selection_functions import get_cv_class


def select_param(train_final,
                 target_column,
                 model_instance,
                 model_name,
                 classification_prediction,
                 cv,
                 weight_column,
                 output_folder):

    x = train_final.drop(columns=[target_column] + ([weight_column] if weight_column else []))
    y = train_final[target_column]
    weight = train_final[weight_column].values.flatten() if weight_column else None
    cv = get_cv_class(cv_method=cv, splits=None, repeats=None)

    if len(model_name) == 1:
        param_grid = ModelGrids.get_grid(model_name[0])
    else:
        param_grid = get_model_grid(model_instance)

    if isinstance(model_instance, LinearRegression):
        print('No hyperparameters in Linear Regression. Skipping hyperparameter optimisation.')
        return model_instance

    if classification_prediction is not None:
        if isinstance(model_instance,
                      (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier)):
            if isinstance(model_instance, DecisionTreeClassifier):
                model_instance.set_params(max_depth=10)
                best_params = rand_search(model_instance=model_instance,
                                          param_grid=param_grid,
                                          cv=cv,
                                          scoring="accuracy",
                                          n_jobs=-1,
                                          weight=weight,
                                          x=x,
                                          y=y)
            else:
                model_instance.set_params(n_estimators=10, n_jobs=-1)
                best_params = perform_grid_search(model_instance=model_instance,
                                                  param_grid=param_grid,
                                                  cv=cv,
                                                  scoring="accuracy",
                                                  n_jobs=-1,
                                                  weight=weight,
                                                  x=x,
                                                  y=y)
        else:
            best_params = perform_grid_search(model_instance=model_instance,
                                              param_grid=param_grid,
                                              cv=cv,
                                              scoring="accuracy",
                                              n_jobs=-1,
                                              weight=weight,
                                              x=x,
                                              y=y)

    else:
        if isinstance(model_instance,
                      (RandomForestRegressor, ExtraTreesRegressor, DecisionTreeRegressor)):
            if isinstance(model_instance, DecisionTreeRegressor):
                model_instance.set_params(max_depth=10)
                best_params = rand_search(model_instance=model_instance,
                                          param_grid=param_grid,
                                          cv=cv,
                                          scoring="r2",
                                          n_jobs=-1,
                                          weight=weight,
                                          x=x,
                                          y=y)
            else:
                model_instance.set_params(n_estimators=10, n_jobs=-1)
                best_params = perform_grid_search(model_instance=model_instance,
                                                  param_grid=param_grid,
                                                  cv=cv,
                                                  scoring="r2",
                                                  n_jobs=-1,
                                                  weight=weight,
                                                  x=x,
                                                  y=y)
        else:
            best_params = perform_grid_search(model_instance=model_instance,
                                              param_grid=param_grid,
                                              cv=cv,
                                              scoring="r2",
                                              n_jobs=-1,
                                              weight=weight,
                                              x=x,
                                              y=y)

    best_model = model_instance.set_params(**best_params)
    best_model.fit(x, y, sample_weight=weight)

    model_filename = os.path.join(output_folder, 'final_model.pkl')
    joblib.dump(best_model, model_filename)

    # coeffs
    if hasattr(best_model, 'coef_'):
        coefficients = best_model.coef_
        coefficients = np.squeeze(coefficients)

        if coefficients.ndim == 1:
            coeff_df = pd.DataFrame({
                'Feature': x.columns,
                'Coefficient': coefficients
            })
        else:
            coeff_df = pd.DataFrame(coefficients.T, columns=x.columns)
            coeff_df.insert(0, 'Feature', x.columns)

        coeff_df.to_csv(os.path.join(output_folder, 'final_model_coefficients.csv'), index=False)

    return best_model


def rand_search(model_instance,
                param_grid,
                cv,
                scoring,
                n_jobs,
                weight,
                x,
                y):
    rand_search = RandomizedSearchCV(model_instance,
                                     param_grid,
                                     cv=cv,
                                     scoring=scoring,
                                     verbose=2,
                                     n_jobs=n_jobs,
                                     n_iter=10,
                                     return_train_score=False,
                                     pre_dispatch='1*n_jobs')
    gc.collect()
    rand_search.fit(x, y, sample_weight=weight)
    best_params = rand_search.best_params_
    print('Best parameters for model are:')
    print(best_params)
    print('CV results:')
    print(rand_search.cv_results_)
    return best_params


def perform_grid_search(model_instance,
                        param_grid,
                        cv,
                        scoring,
                        n_jobs,
                        weight,
                        x,
                        y):
    grid_search = GridSearchCV(model_instance,
                               param_grid,
                               cv=cv,
                               scoring=scoring,
                               verbose=2,
                               n_jobs=n_jobs,
                               return_train_score=False,
                               pre_dispatch='1*n_jobs')
    gc.collect()
    grid_search.fit(x, y, sample_weight=weight)
    best_params = grid_search.best_params_
    print('Best parameters for model are:')
    print(best_params)
    print('CV results:')
    print(grid_search.cv_results_)
    return best_params

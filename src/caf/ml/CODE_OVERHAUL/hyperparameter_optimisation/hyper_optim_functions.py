# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position

import gc
import os
from multiprocessing import cpu_count
import joblib
import pandas as pd
from caf.ml.CODE_OVERHAUL.MODELS.NorCom.norcom_temporary_inputs import ParamGridStorage
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# todo move and refine hyperparm funcs


def modified_hyper_optimisation(model, data, target_column, output_folder, weight_column,
                                original_training_data, index_columns):
    print('Hyperparameter optimisation beginning')
    if isinstance(model, LogisticRegression):
        model.set_params(max_iter=1000)

    if not data.index.names == index_columns:
        data = data.set_index(index_columns)

    if weight_column in data.columns:
        data = data.drop(columns=weight_column)

    x = data.drop(columns=[target_column])
    y = data[target_column]
    cv = TimeSeriesSplit(n_splits=3)

    weight_df = original_training_data[weight_column]
    weight = weight_df.values.flatten()

    n_cores = cpu_count()
    n_jobs = max(1, n_cores - 1)

    def rand_search(model_instance, param_grid, cv, scoring):
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

    param_grid_storage = ParamGridStorage()
    print(model)
    if isinstance(model, GradientBoostingClassifier):
        param_grid = param_grid_storage.gb_params
    elif isinstance(model, RandomForestClassifier):
        param_grid = param_grid_storage.rf_params
    elif isinstance(model, DecisionTreeClassifier):
        param_grid = param_grid_storage.dt_params
    elif isinstance(model, OneVsRestClassifier) and isinstance(model.estimator, LinearSVC):
        param_grid = param_grid_storage.svm_params
    elif isinstance(model, LogisticRegression):
        if model.get_params()['penalty'] == 'l1' and model.get_params()['solver'] == 'liblinear':
            param_grid = param_grid_storage.logit_l1_params
        elif model.get_params()['penalty'] == 'l2':
            param_grid = param_grid_storage.logit_l2_params
        elif model.get_params()['penalty'] == 'elasticnet':
            param_grid = param_grid_storage.logit_elastic_net_params
        elif model.get_params()['multi_class'] == 'multinomial':
            param_grid = param_grid_storage.logit_multinomial_params
        else:
            param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                          "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                          "max_iter": [1000, 2000, 3000]}
    elif isinstance(model, LinearSVC):
        param_grid = param_grid_storage.svm_binary_params
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    best_params = rand_search(model, param_grid, cv=cv, scoring="accuracy")
    best_model = model.set_params(**best_params)
    best_model.fit(x, y, sample_weight=weight)

    model_filename = os.path.join(output_folder, 'cafml_final_model.pkl')
    joblib.dump(best_model, model_filename)
    print(f"Model saved to: {model_filename}")

    # coeffs
    if hasattr(best_model, 'coef_'):
        coefficients = best_model.coef_
        if coefficients.ndim == 1:
            coeff_df = pd.DataFrame({'Feature': x.columns, 'Coefficient': coefficients})
        else:
            coeff_df = pd.DataFrame(coefficients.T,
                                    columns=[f'Class_{i}' for i in range(coefficients.shape[0])])
            coeff_df['Feature'] = x.columns
        coeff_df.to_csv(os.path.join(output_folder, 'final_model_coefficients.csv'), index=False)
        print(f"Coefficients saved to: {os.path.join(output_folder, 'final_model_coefficients.csv')}")
    else:
        print("Model does not have coefficients attribute.")

    print('Hyperparameter optimisation finished')
    return best_model

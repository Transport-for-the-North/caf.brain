# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
import joblib
import pandas as pd
from caf.ml.CODE_OVERHAUL.MODELS.NorCom.norcom_temporary_inputs import ModelStorage, ParamGridStorage
from caf.ml.old_inputs.cafml_inputs import Models
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

# todo move and refine model selection / creation funcs

def model_prep(training_df, target_column, output_folder, weight_column, model_to_use):
    print('Model prep beginning')

    x = training_df.drop(columns=[target_column])
    y = training_df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    weight = None
    if weight_column in training_df.columns:
        weight_df = x_train[weight_column]
        weight = weight_df.values.flatten()
        weight_df.to_csv(os.path.join(output_folder, 'weight.csv'))
        x_train = x_train.drop(columns=weight_column)
        x_test = x_test.drop(columns=weight_column)

    print(x_test)
    print(x_test.columns)
    model_storage = ModelStorage()
    param_grid_storage = ParamGridStorage()

    models = {
        'gb': {'model': model_storage.gb, 'filename': 'gb_basic_cafml_modelfit.pkl',
               'params': param_grid_storage.gb_params},
        'rf': {'model': model_storage.rf, 'filename': 'rf_basic_cafml_modelfit.pkl',
               'params': param_grid_storage.rf_params},
        'dt': {'model': model_storage.dt, 'filename': 'dt_basic_cafml_modelfit.pkl',
               'params': param_grid_storage.dt_params},
        'svm': {'model': model_storage.svm, 'filename': 'svm_basic_cafml_modelfit.pkl',
                'params': param_grid_storage.svm_params},
        'logistic': {'model': LogisticRegression(penalty='elasticnet', solver='saga',
                                                 multi_class='multinomial',
                                                 l1_ratio=0.5, n_jobs=-1, max_iter=1000),
                     'filename': 'logistic_basic_cafml_modelfit.pkl',
                     'params': {'C': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}},
        'svm_binary': {'model': model_storage.svm_binary, 'filename': 'svm_binary_basic_cafml_modelfit.pkl',
                       'params': param_grid_storage.svm_binary_params},
        'logit_l1': {'model': model_storage.logit_l1, 'filename': 'logit_l1_basic_cafml_modelfit.pkl',
                     'params': param_grid_storage.logit_l1_params},
        'logit_l2': {'model': model_storage.logit_l2,
                     'filename': 'logit_l2_basic_cafml_modelfit.pkl',
                     'params': param_grid_storage.logit_l2_params},
        'logit_elastic_net': {'model': model_storage.logit_elastic_net,
                              'filename': 'logit_elastic_net_basic_cafml_modelfit.pkl',
                              'params': param_grid_storage.logit_elastic_net_params},
        'logit_multinomial': {'model': model_storage.logit_multinomial,
                              'filename': 'logit_multinomial_basic_cafml_modelfit.pkl',
                              'params': param_grid_storage.logit_multinomial_params}
    }

    if model_to_use not in models or model_to_use is None:
        print(f"Invalid model specified: {model_to_use}. Defaulting to logistic regression.")
        model_to_use = 'logistic'

    model_filename = os.path.join(output_folder, models[model_to_use]['filename'])
    if os.path.exists(model_filename):
        print(f"Loading existing {model_to_use} model")
        model = models[model_to_use]['model']
        model_fit = joblib.load(model_filename)
    else:
        model = models[model_to_use]['model']
        model_fit = model.fit(x_train, y_train, sample_weight=weight)
        joblib.dump(model_fit, model_filename)

    y_pred = model_fit.predict(x_test)
    residuals = y_test - y_pred

    if model_to_use in ['svm', 'logistic', 'logit_l1', 'logit_l2',
                        'logit_elastic_net', 'logit_multinomial']:
        if hasattr(model_fit, 'coef_'):
            coefficients = model_fit.coef_
            print(f"Model Coefficients shape: {coefficients.shape}")

            coeff_df = pd.DataFrame(coefficients, columns=x_train.columns)
            coeff_df = coeff_df.T
            coeff_df.columns = [f'Class {i}' for i in range(coeff_df.shape[1])]
            coeff_df.reset_index(inplace=True)
            coeff_df.rename(columns={'index': 'Feature'}, inplace=True)
            coeff_df.to_csv(os.path.join(output_folder, 'initial_model_coefficients.csv'),
                            index=False)


    print('Model prep ending')
    return model, residuals


def select_model(x: pd.DataFrame, y: pd.DataFrame, models_to_test: list[Models], output_folder):

    acc = {}
    score = 0
    return_model = None
    for model_enum in models_to_test:
        model_instance = model_enum.value()
        model_name = model_enum.name.lower()

        scores_r2 = cross_val_score(model_instance, x, y, cv=5, scoring="r2", n_jobs=-1)
        scores_mse = -cross_val_score(model_instance, x, y, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

        mean_r2 = scores_r2.mean()
        mean_mse = scores_mse.mean()

        acc[model_name] = {'R-squared': mean_r2, 'MSE': mean_mse}
        if mean_r2 > score:
            score = mean_r2
            return_model = model_instance
    print(f"Best model score: {score}")
    evaluation_df = pd.DataFrame.from_dict(acc, orient='index')

    output_filename = 'pre_transformation_model_evaluation_results.csv'
    output_path = os.path.join(output_folder, output_filename)
    evaluation_df.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"Pre-transformation model evaluation results exported to: {output_path}")

    return return_model


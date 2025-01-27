# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from caf.ml.MODELS.prediction_model.prediction_model_inputs import Models
from sklearn.model_selection import train_test_split, cross_val_score


def find_coefs(train,
               target_column,
               output_folder,
               weight_column,
               model_initialised):
    x = train.drop(columns=[target_column])
    y = train[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

    weight = None
    if weight_column in train.columns:
        weight_df = x_train[weight_column]
        weight = weight_df.values.flatten()
        x_train = x_train.drop(columns=weight_column)
        x_test = x_test.drop(columns=weight_column)


    model_filename = os.path.join(output_folder, 'initial_fitted_model.pkl')
    if os.path.exists(model_filename):
        model_fit = joblib.load(model_filename)
    else:
        model_fit = model_initialised.fit(x_train, y_train, sample_weight=weight)
        joblib.dump(model_fit, model_filename)

    y_pred = model_fit.predict(x_test)
    residuals = y_test - y_pred


    if hasattr(model_fit, 'coef_'):
        coefficients = model_fit.coef_
        print(f"Model Coefficients shape: {coefficients.shape}")

        coefficients = np.squeeze(coefficients)

        if coefficients.ndim == 1:
            coeff_df = pd.DataFrame({
                'Feature': x_train.columns,
                'Coefficient': coefficients
            })
        else:
            coeff_df = pd.DataFrame(coefficients.T, columns=x_train.columns)
            coeff_df.insert(0, 'Feature', x_train.columns)
            # coeff_df = pd.DataFrame(coefficients.T, columns=x_train.columns[:coefficients.shape[1]])
            # coeff_df.insert(0, 'Feature', x_train.columns[:coefficients.shape[1]])

        coeff_df.to_csv(os.path.join(output_folder, 'initial_model_coefficients.csv'), index=False)


    return model_fit, residuals, x_train, x_test, y_train, y_test


def select_model(train: pd.DataFrame,
                 target_column: str,
                 weight_column: str,
                 models_to_test: list[Models],
                 output_folder: Path,
                 binary_prediction: str):

    weight = None
    y = train[target_column]
    x = train.drop(columns=target_column)
    if weight_column in train.columns:
        weight = x[weight_column]
        x = x.drop(columns=weight_column)

    acc = {}
    best_score = float('-inf')
    best_model = None

    for model_enum in models_to_test:
        model_instance = model_enum.get_model()
        print(model_instance)

        if isinstance(model_instance, LogisticRegression):
            model_instance.set_params(max_iter=1000)

        if binary_prediction is not None:
            scores_r2, scores_mse = score_regression(weight=weight,
                                                     model_instance=model_instance,
                                                     x=x,
                                                     y=y)
            mean_score = scores_r2.mean()
            acc[model_enum] = {'R-squared': scores_r2.mean(), 'MSE': scores_mse.mean()}

        else:
            scores_f1, scores_auc = score_classification(weight=weight,
                                                         model_instance=model_instance,
                                                         x=x,
                                                         y=y)
            mean_score = scores_f1.mean()
            acc[model_enum] = {'F1': scores_f1.mean(), 'AUC': -scores_auc.mean()}

        if mean_score > best_score:
            best_score = mean_score
            best_model = model_enum.get_model()

    print(f"Best model: {best_model}")
    print(f"Best model score: {best_score}")
    evaluation_df = pd.DataFrame.from_dict(acc, orient='index')

    output_filename = 'model_algorithm_evaluation.csv'
    output_path = os.path.join(output_folder, output_filename)
    evaluation_df.to_csv(output_path, index=True)

    return best_model


def score_regression(weight,
                     model_instance,
                     x,
                     y):
    if weight is not None:
        scores_r2 = cross_val_score(model_instance, x, y, cv=3,
                                    scoring="r2", n_jobs=-1,
                                    fit_params={'sample_weight': weight}, verbose=1)
        scores_mse = -cross_val_score(model_instance, x, y, cv=3,
                                      scoring="neg_mean_squared_error",
                                      n_jobs=-1, fit_params={'sample_weight': weight}, verbose=1)
    else:
        scores_r2 = cross_val_score(model_instance, x, y, cv=3,
                                    scoring="r2", n_jobs=-1, verbose=1)
        scores_mse = -cross_val_score(model_instance, x, y, cv=3,
                                      scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

    return scores_r2, scores_mse

def score_classification(weight,
                         model_instance,
                         x,
                         y):
    if weight is not None:
        scores_f1 = cross_val_score(model_instance, x, y, cv=3,
                                    scoring="f1", n_jobs=-1,
                                    fit_params={'sample_weight': weight}, verbose=1)
        scores_auc = -cross_val_score(model_instance, x, y, cv=3,
                                      scoring="roc_auc",
                                      n_jobs=-1, fit_params={'sample_weight': weight}, verbose=1)
    else:
        scores_f1 = cross_val_score(model_instance, x, y, cv=3,
                                    scoring="f1", n_jobs=-1, verbose=1)
        scores_auc = -cross_val_score(model_instance, x, y, cv=3,
                                      scoring="roc_auc", n_jobs=-1, verbose=1)

    return scores_f1, scores_auc

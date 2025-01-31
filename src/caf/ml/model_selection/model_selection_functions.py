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
from sklearn.metrics import log_loss, mean_squared_error
from caf.ml.MODELS.prediction_model.prediction_model_inputs import Models
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import stats


def initialise_model(train,
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
        if weight is not None:
            model_fit = model_initialised.fit(x_train, y_train, sample_weight=weight)
        else:
            model_fit = model_initialised.fit(x_train, y_train)

        joblib.dump(model_fit, model_filename)

    y_pred = model_fit.predict(x_test)
    residuals = y_test - y_pred

    # old method
    # coeff_df = None
    # if hasattr(model_fit, 'coef_'):
    #     coefficients = model_fit.coef_
    #     coefficients = np.squeeze(coefficients)
    #
    #     if coefficients.ndim == 1:
    #         coeff_df = pd.DataFrame({
    #             'Feature': x_train.columns,
    #             'Coefficient': coefficients
    #         })
    #     else:
    #         coeff_df = pd.DataFrame(coefficients.T, columns=x_train.columns)
    #         coeff_df.insert(0, 'Feature', x_train.columns)
    #         # coeff_df = pd.DataFrame(coefficients.T, columns=x_train.columns[:coefficients.shape[1]])
    #         # coeff_df.insert(0, 'Feature', x_train.columns[:coefficients.shape[1]])

    coeff_df, mse = calculate_model_coeff(model=model_fit,
                                          x_train=x_train,
                                          x_test=x_test,
                                          y_test=y_test,
                                          residuals=residuals)

    if coeff_df is not None:
        coeff_df.to_csv(os.path.join(output_folder, 'initial_model_coefficients.csv'), index=False)

    return model_fit, residuals, x_train, x_test, y_train, y_test, mse


def select_model(train: pd.DataFrame,
                 target_column: str,
                 weight_column: str,
                 models_to_test: list[Models],
                 output_folder: Path,
                 classification_prediction: str):

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
        # scikit
        model_instance = model_enum.get_model()
        print(f"Testing model: {model_instance}")

        if isinstance(model_instance, LogisticRegression):
            model_instance.set_params(max_iter=1000)

        if classification_prediction is not None:
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


def calculate_model_coeff(model,
                          x_train,
                          x_test,
                          y_test,
                          residuals):

    if not hasattr(model, 'coef_'):
        return None

    n = x_train.shape[0]
    p = x_train.shape[1]
    dof = n - p - 1

    is_classifier = hasattr(model, 'predict_proba')
    if is_classifier:
        # classification
        proba = model.predict_proba(x_test)
        if proba.shape[1] == 2:
            mse = log_loss(y_test, proba[:, 1])
        else:
            mse = log_loss(y_test, proba)
    else:
        # regression
        mse = np.mean(residuals ** 2)

    # variance-covariance matrix
    X_with_intercept = np.column_stack([np.ones(n), x_train]) if hasattr(model,
                                                                         'intercept_') else x_train
    covariance_matrix = np.linalg.pinv(X_with_intercept.T.dot(X_with_intercept)) * mse

    std_errors = np.sqrt(np.diag(covariance_matrix))

    if hasattr(model, 'intercept_'):
        intercept = np.array(model.intercept_).flatten()
        coefficients = np.array(model.coef_).flatten()
        coefficients = np.concatenate([intercept, coefficients])
    else:
        coefficients = model.coef_.flatten()

    # t-values and p-values
    t_values = coefficients / std_errors
    p_values = 2 * (1 - stats.t.cdf(abs(t_values), dof))

    feature_names = ['intercept'] + list(x_train.columns) if hasattr(model, 'intercept_') else list(x_train.columns)

    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Std_Error': std_errors,
        'T_Value': t_values,
        'P_Value': p_values
    })

    return coeff_df, mse


def calculate_final_coefficients(model,
                                 test_data,
                                 training_mse,
                                 predictions,
                                 validation_data,
                                 target_column,
                                 is_classification):
    if not hasattr(model, 'coef_'):
        return None

    if validation_data is not None and target_column is not None:
        if is_classification:
            # classification
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(test_data)
                if proba.shape[1] == 2:
                    # Binary
                    mse = log_loss(validation_data[target_column], proba[:, 1])
                else:
                    # Multiclass
                    mse = log_loss(validation_data[target_column], proba)
            else:
                # LinearSVC
                mse = log_loss(validation_data[target_column],
                               predictions,
                               labels=np.unique(validation_data[target_column]))
            print(f"Using validation log loss: {mse}")
        else:
            # regression
            mse = mean_squared_error(validation_data[target_column], predictions)
            print(f"Using validation MSE: {mse}")
    else:
        mse = training_mse
        print(f"Using training {'log loss' if is_classification else 'MSE'}: {mse}")


    feature_names = list(test_data.columns)
    if hasattr(model, 'intercept_'):
        coefficients = np.array(model.coef_).flatten()
        intercept = np.array(model.intercept_).flatten()
        coefficients = np.concatenate([intercept, coefficients])
        feature_names = ['intercept'] + feature_names
    else:
        coefficients = model.coef_.flatten()

    n = test_data.shape[0]
    p = test_data.shape[1]
    dof = n - p - 1

    X_with_intercept = np.column_stack([np.ones(n), test_data]) if hasattr(model,
                                                                           'intercept_') else test_data
    covariance_matrix = np.linalg.pinv(X_with_intercept.T.dot(X_with_intercept)) * mse
    std_errors = np.sqrt(np.diag(covariance_matrix))
    t_values = coefficients / std_errors
    p_values = 2 * (1 - stats.t.cdf(abs(t_values), dof))

    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Std_Error': std_errors,
        'T_Value': t_values,
        'P_Value': p_values,
        'MSE_Source': 'validation' if validation_data is not None else 'training'
    })

    return coeff_df

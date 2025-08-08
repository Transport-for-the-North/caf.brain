# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# Built-Ins
import logging

# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
from pathlib import Path
from typing import List

# Third Party
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

# Local Imports
from caf.brain.ml.main_models.prediction_model.inputs import Models

LOG = logging.getLogger(__name__)


def initialise_model(
    train: pd.DataFrame,
    target_column: str,
    output_folder: Path,
    weight_column: str,
    model_initialised,
    classification_prediction: tuple[int, ...],
):
    """
    Fits the initialised model with machine learning prediction convention.
    This gives a first look into how well the model will preform prior to
    entering the machine learning pipeline.

    :param train: processed input data split into train subset.
    :param target_column: String column name of value to predict.
    :param output_folder: Path to output location.
    :param weight_column: Optional string column value to be used as weight.
    :param model_initialised: Initialised SciKitLearn model.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.
    :return:
        model_fit: Fitted model on train_test_split test data.
        residuals: Truth values form the train_test_split against the predictions.
        x_train: Series of train data to be used as train.
        x_test: Series of test data to be used as unseen test data.
        y_train: Series of target column inside train to be used as train.
        y_test: Series of target column inside train to be used as validation for
                predictions.
        mse: Mean squared error of predictions.
    """
    x = train.drop(columns=[target_column])
    y = train[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

    weight = None
    if weight_column in train.columns:
        weight_df = x_train[weight_column]
        weight = weight_df.values.flatten()
        x_train = x_train.drop(columns=weight_column)
        x_test = x_test.drop(columns=weight_column)

    model_filename = os.path.join(output_folder, "initial_fitted_model.pkl")
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

    coeff_df, mse = calculate_model_coeff(
        model=model_fit,
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        residuals=residuals,
        classification_prediction=classification_prediction,
        y_pred=y_pred,
    )

    if coeff_df is not None:
        coeff_df.to_csv(
            os.path.join(output_folder, "initial_model_coefficients.csv"), index=False
        )

    return model_fit, residuals, x_train, x_test, mse


def select_model(
    train: pd.DataFrame,
    target_column: str,
    weight_column: str,
    models_to_test: List[Models],
    output_folder: Path,
    classification_prediction: tuple[int, ...],
):
    """
    Function to quickly assess the best model for the data based on the list of
    provided models.

    :param train: processed input data split into train subset.
    :param target_column: sting column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param models_to_test: List or one algorithm to use as the base of the model.
                           Available algorithms can be seen in
                           prediction_model_inputs.py or __info__.py.
    :param output_folder: Path to output location.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.

    :return: Best performing model initialised.
    """
    weight = None
    y = train[target_column]
    x = train.drop(columns=target_column)
    if weight_column in train.columns:
        weight = x[weight_column]
        x = x.drop(columns=weight_column)

    acc = {}
    best_score = float("-inf")
    best_model = None

    for model_enum in models_to_test:
        # scikit
        model_instance = model_enum.get_model()
        LOG.info(f"Testing model: {model_instance}")

        if isinstance(model_instance, LogisticRegression):
            model_instance.set_params(max_iter=1000)

        if classification_prediction:
            scores_f1, scores_auc = score_classification(
                weight=weight, model_instance=model_instance, x=x, y=y
            )
            mean_score = scores_f1.mean()
            acc[model_enum] = {"F1": scores_f1.mean(), "AUC": -scores_auc.mean()}
        else:
            scores_r2, scores_mse = score_regression(
                weight=weight, model_instance=model_instance, x=x, y=y
            )
            mean_score = scores_r2.mean()
            acc[model_enum] = {"R-squared": scores_r2.mean(), "MSE": scores_mse.mean()}

        if mean_score > best_score:
            best_score = mean_score
            best_model = model_enum.get_model()

    LOG.info(f"Best model: {best_model}")
    LOG.info(f"Best model score: {best_score}")
    evaluation_df = pd.DataFrame.from_dict(acc, orient="index")

    output_filename = "model_algorithm_evaluation.csv"
    output_path = os.path.join(output_folder, output_filename)
    evaluation_df.to_csv(output_path, index=True)

    return best_model


def score_regression(
    weight: pd.DataFrame, model_instance: Models, x: pd.DataFrame, y: pd.DataFrame
):
    """
    Function to score regression based problems.

    :param weight: Pandas dataframe of weight values from the original
                   train input data.
    :param model_instance: Initialised model.
    :param x: Train data split into only the explanatory variables. Target
              and weight should be removed. Any index columns should be
              set.
    :param y: Train data split into only the target. Any index columns should
              be set.

    :return:
        scores_r2: Series of R2 scores.
        scores_mse: Series of mean squared error scores.
    """
    if weight is not None:
        scores_r2 = cross_val_score(
            model_instance,
            x,
            y,
            cv=3,
            scoring="r2",
            n_jobs=-1,
            fit_params={"sample_weight": weight},
            verbose=1,
        )
        scores_mse = -cross_val_score(
            model_instance,
            x,
            y,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            fit_params={"sample_weight": weight},
            verbose=1,
        )
    else:
        scores_r2 = cross_val_score(
            model_instance, x, y, cv=3, scoring="r2", n_jobs=-1, verbose=1
        )
        scores_mse = -cross_val_score(
            model_instance, x, y, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
        )

    return scores_r2, scores_mse


def score_classification(
    weight: pd.DataFrame, model_instance: Models, x: pd.DataFrame, y: pd.DataFrame
):
    """
    Function to score classification based problems.

    :param weight: Pandas dataframe of weight values from the original
                   train input data.
    :param model_instance: Initialised model.
    :param x: Train data split into only the explanatory variables. Target
              and weight should be removed. Any index columns should be
              set.
    :param y: Train data split into only the target. Any index columns should
              be set.

    :return:
        scores_f1: Series of F1 scores.
        scores_auc: Series of AUC scores.
    """
    if weight is not None:
        scores_f1 = cross_val_score(
            model_instance,
            x,
            y,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            fit_params={"sample_weight": weight},
            verbose=1,
        )
        scores_auc = -cross_val_score(
            model_instance,
            x,
            y,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            fit_params={"sample_weight": weight},
            verbose=1,
        )
    else:
        scores_f1 = cross_val_score(
            model_instance, x, y, cv=3, scoring="f1", n_jobs=-1, verbose=1
        )
        scores_auc = -cross_val_score(
            model_instance, x, y, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
        )

    return scores_f1, scores_auc


def calculate_model_coeff(
    model,
    x_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
    residuals: pd.Series,
    classification_prediction: tuple[int, ...],
    y_pred: pd.Series,
):
    """
    Calculates models linear coefficents if applicable to model selected.

    :param model: Fitted model on train_test_split of training data.
    :param x_train: Series of train data to be used as train.
    :param x_test: Series of test data to be used as unseen test data.
    :param y_test: Series of target column inside train to be used as validation
                   for predictions.
    :param residuals: Series of residual values based on x_test predictions.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.
    :param y_pred: Series of predicted values based on training data.

    :return:
        coeff_df: Dataframe of coefficient values and other relevant statistics.
        mse: Mean squared error of predictions.
    """

    if not hasattr(model, "coef_"):
        return None, None

    n = x_train.shape[0]
    p = x_train.shape[1]
    dof = n - p - 1

    if classification_prediction:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_test)
            if proba.shape[1] == 2:
                # binary
                mse = log_loss(y_test, proba[:, 1])
            else:
                # multiclass
                mse = log_loss(y_test, proba)
        else:
            # linear svc
            mse = log_loss(y_test, y_pred, labels=np.unique(y_test))
    else:
        # regression
        mse = np.mean(residuals**2)

    # variance-covariance matrix
    X_with_intercept = (
        np.column_stack([np.ones(n), x_train]) if hasattr(model, "intercept_") else x_train
    )
    covariance_matrix = np.linalg.pinv(X_with_intercept.T.dot(X_with_intercept)) * mse

    std_errors = np.sqrt(np.diag(covariance_matrix))

    if hasattr(model, "intercept_"):
        intercept = np.array(model.intercept_).flatten()
        coefficients = np.array(model.coef_).flatten()
        coefficients = np.concatenate([intercept, coefficients])
    else:
        coefficients = model.coef_.flatten()

    # t-values and p-values
    t_values = coefficients / std_errors
    p_values = 2 * (1 - stats.t.cdf(abs(t_values), dof))

    feature_names = (
        ["intercept"] + list(x_train.columns)
        if hasattr(model, "intercept_")
        else list(x_train.columns)
    )

    coeff_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Std_Error": std_errors,
            "T_Value": t_values,
            "P_Value": p_values,
        }
    )

    return coeff_df, mse


def calculate_final_coefficients(
    model,
    test_data: pd.DataFrame,
    training_mse: pd.Series,
    predictions: pd.Series,
    validation_data: pd.DataFrame,
    target_column: str,
    is_classification: tuple[int, ...],
    drop_vals: pd.DataFrame,
    cols_dropped_by_feat_select: pd.DataFrame,
):
    """
    Calculates models linear coefficents if applicable to model used for
    prediction.

    :param model: Fitted final model for prediction on unseen (test) data.
    :param test_data: Dataframe of final test data post feature selection.
    :param training_mse: Mean squared error of predictions based on
                         training data.
    :param predictions: Predicted values based on the test data and set to the
                        same index.
    :param validation_data: Validation data if available.
    :param target_column: String column name of value to predict.
    :param is_classification: List of integers that correspond to the
                              target column. The value(s) to predict
                              in a classification problem.
    :param drop_vals: Values dropped during encoding of categorical variables.
    :param cols_dropped_by_feat_select: These are the columns removed due to
                                        feature selection.

    :return:
        coeff_df: Dataframe of coefficient values and other relevant statistics.
    """
    if not hasattr(model, "coef_"):
        return None

    if validation_data is not None and target_column is not None:
        if is_classification:
            # classification
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(test_data)
                if proba.shape[1] == 2:
                    # Binary
                    mse = log_loss(validation_data[target_column], proba[:, 1])
                else:
                    # Multiclass
                    mse = log_loss(validation_data[target_column], proba)
            else:
                # LinearSVC
                mse = log_loss(
                    validation_data[target_column],
                    predictions,
                    labels=np.unique(validation_data[target_column]),
                )
            LOG.info(f"Using validation log loss: {mse}")
        else:
            # regression
            mse = mean_squared_error(validation_data[target_column], predictions)
            LOG.info(f"Using validation MSE: {mse}")

    else:
        mse = training_mse
        LOG.info(f"Using training {'log loss' if is_classification else 'MSE'}: {mse}")

    feature_names = list(test_data.columns)
    if hasattr(model, "intercept_"):
        coefficients = np.array(model.coef_).flatten()
        intercept = np.array(model.intercept_).flatten()
        coefficients = np.concatenate([intercept, coefficients])
        feature_names = ["intercept"] + feature_names
    else:
        coefficients = model.coef_.flatten()

    n = test_data.shape[0]
    p = test_data.shape[1]
    dof = n - p - 1

    X_with_intercept = (
        np.column_stack([np.ones(n), test_data]) if hasattr(model, "intercept_") else test_data
    )
    covariance_matrix = np.linalg.pinv(X_with_intercept.T.dot(X_with_intercept)) * mse
    std_errors = np.sqrt(np.diag(covariance_matrix))
    t_values = coefficients / std_errors
    p_values = 2 * (1 - stats.t.cdf(abs(t_values), dof))

    coeff_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Std_Error": std_errors,
            "T_Value": t_values,
            "P_Value": p_values,
            "MSE_Source": "validation" if validation_data is not None else "training",
        }
    )

    if drop_vals is not None:
        # df with extra columns
        drop_vals_features = pd.DataFrame(
            {
                "Feature": drop_vals.columns,
                "Coefficient": "N/A",
                "Std_Error": "N/A",
                "T_Value": "N/A",
                "P_Value": "N/A",
                "MSE_Source": ["dropped during encoding"] * drop_vals.shape[1],
            }
        )
    else:
        drop_vals_features = None

    if cols_dropped_by_feat_select is not None:
        feat_select_features = pd.DataFrame(
            {
                "Feature": cols_dropped_by_feat_select.columns,
                "Coefficient": "N/A",
                "Std_Error": "N/A",
                "T_Value": "N/A",
                "P_Value": "N/A",
                "MSE_Source": ["dropped during feature selection"]
                * cols_dropped_by_feat_select.shape[1],
            }
        )
    else:
        feat_select_features = None

    if (
        drop_vals_features is not None
        and len(drop_vals_features) > 0
        and feat_select_features is not None
        and len(feat_select_features) > 0
    ):
        coeff_df = pd.concat(
            [coeff_df, drop_vals_features, feat_select_features], ignore_index=True
        )
    elif drop_vals_features is not None and len(drop_vals_features) > 0:
        coeff_df = pd.concat([coeff_df, drop_vals_features], ignore_index=True)
    elif feat_select_features is not None and len(feat_select_features) > 0:
        coeff_df = pd.concat([coeff_df, feat_select_features], ignore_index=True)

    return coeff_df

# -*- coding: utf-8 -*-
"""
Created on: 1/17/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from pathlib import Path
from typing import List
import pandas as pd
from caf.brain.ml.data_analysis.data_analysis_functions import pre_forecast_data_analysis


def main_evaluate_input_data(
    model_fit,
    model_initialised,
    residuals: pd.Series,
    x_test: pd.Series,
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    full_transformations: bool,
    train_unscaled: pd.DataFrame,
    test_unscaled: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    target_column: str,
    weight_column: str,
    x_train: pd.Series,
    output_folder: Path,
    is_time_series: bool,
    numerical_pipeline,
):
    """
    Main function for testing data quality

    :param model_fit: Fitted model on train_test_split test data.
    :param model_initialised: Initialised SciKitLearn model.
    :param residuals: Truth values form the train_test_split against the predictions.
    :param x_test: Series of test data to be used as unseen test data.
    :param train_scaled: Processed input data split into train set.
    :param test_scaled: Processed input data split into train set.
    :param full_transformations: If true then transformations will be applied
                                 to the continuous data inside of train and
                                 test scaled. This fixes any potential data
                                 issues present.
    :param train_unscaled: Input data unprocessed split into train.
    :param test_unscaled: Input data unprocessed split into test.
    :param numerical_features: List of string column names that are
                               continuous variables.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :param target_column: String column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param x_train: Series of train data to be used as train.
    :param output_folder: Path to output location.
    :param is_time_series: If true then data must be time series. Time series
                           based characteristics are taken into consideration
                           during function execution.
    :param numerical_pipeline: Stored numerical transformation pipeline for
                               full model runs. Left as None if not a full
                               model run.

    :return:
        train_transformed: training data with the numerical features transformed
                           or train_scaled if transformations not applied.
        test_transformed: test data with the numerical features transformed
                           or test_scaled if transformations not applied.
    """
    train_transformed, test_transformed = pre_forecast_data_analysis(
        residuals=residuals,
        model=model_fit,
        x_test=x_test,
        train_scaled=train_scaled,
        test_scaled=test_scaled,
        full_transformations=full_transformations,
        train_unscaled=train_unscaled,
        test_unscaled=test_unscaled,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target_column,
        weight_column=weight_column,
        x_train=x_train,
        model_initialised=model_initialised,
        output_folder=output_folder,
        is_time_series=is_time_series,
        numerical_pipeline=numerical_pipeline,
    )

    if train_transformed is not None:
        return train_transformed, test_transformed
    else:
        return train_scaled, test_scaled

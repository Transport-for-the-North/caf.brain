# -*- coding: utf-8 -*-
"""
Created on: 1/24/2025
Original author: Adil Zaheer
"""
# Built-Ins
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from pathlib import Path

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml.prediction.prediction_functions import prediction


def main_prediction(
    model,
    test: pd.DataFrame,
    target_column: str,
    output_folder: Path,
    validation: pd.DataFrame,
    weight_column: str,
    classification_prediction: tuple[int, ...],
    mse: pd.Series,
    drop_vals: pd.DataFrame,
    cols_dropped_by_feat_select: pd.DataFrame,
):
    """
    Main prediction function.

    :param model: Fitted final model for prediction on unseen (test) data.
    :param test: Dataframe of final test data post feature selection.
    :param target_column: String column name of value to predict.
    :param output_folder: Path to output location.
    :param validation: Validation data if available.
    :param weight_column: Optional string column value to be used as weight.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.
    :param mse: Mean squared error or None. Dependency on if the algorithm selected
                has coefficient values.
    :param drop_vals: Values dropped during encoding of categorical variables.
    :param cols_dropped_by_feat_select: These are the columns removed due to
                                        feature selection.

    :return:
        y_pred: Predicted values based on the test data and set to the same
                index.
    """

    prediction(
        model=model,
        test=test,
        target_column=target_column,
        output_folder=output_folder,
        validation=validation,
        weight_column=weight_column,
        classification_prediction=classification_prediction,
        mse=mse,
        drop_vals=drop_vals,
        cols_dropped_by_feat_select=cols_dropped_by_feat_select,
    )

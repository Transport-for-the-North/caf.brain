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
from caf.brain.ml.prediction.functions import prediction


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

    Parameters
    ----------
    model : object
        Fitted final model for prediction on unseen (test) data.
    test : pandas.DataFrame
        Final test data after feature selection.
    target_column : str
        Name of the column to predict.
    output_folder : pathlib.Path
        Path to output location.
    validation : pandas.DataFrame or None
        Validation data, if available.
    weight_column : str
        Optional column name to be used as sample weights.
    classification_prediction : tuple of int or None
        Target values to predict in a classification problem.
    mse : float or None
        Mean squared error, or None if the algorithm selected does not provide coefficients.
    drop_vals : pandas.DataFrame or None
        Values dropped during encoding of categorical variables.
    cols_dropped_by_feat_select : pandas.DataFrame or None
        Columns removed due to feature selection.

    Returns
    -------
    None
        Saves predictions and coefficients to disk.
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

# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# Built-Ins
import logging

# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from pathlib import Path

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml.model_selection.functions import (
    initialise_model,
    select_model,
)

LOG = logging.getLogger(__name__)


def main_model_selection(
    train: pd.DataFrame,
    target_column: str,
    weight_column: str,
    output: Path,
    model,
    classification_prediction: tuple[int, ...],
):
    """
    Main function for selecting a model algorithm and extracting relevant attributes.

    Parameters
    ----------
    train : pandas.DataFrame
        Processed input data split into the training subset.
    target_column : str
        Name of the column to predict.
    weight_column : str
        Optional column name to be used as sample weights.
    output : pathlib.Path
        Path to the output location.
    model : list or object
        List or single algorithm to use as the base of the model. Available algorithms
        can be seen in `prediction_model_inputs.py` or `__info__.py`.
    classification_prediction : tuple of int
        Target values to predict in a classification problem.

    Returns
    -------
    model_initialised : object
        Initialised model algorithm from the Models enum class.
    model_fit : object
        Model fit on training data.
    residuals : pandas.Series
        Difference between true and predicted values (based on training data).
    x_test : pandas.DataFrame
        Test features from train/test split.
    x_train : pandas.DataFrame
        Training features from train/test split.
    mse : float or None
        Mean squared error, or None if the algorithm does not provide coefficients.

    Raises
    ------
    ValueError
        If the model is not provided or is incorrectly specified.
    """

    if not isinstance(model, list):
        model = [model]

    if len(model) == 1:
        model_initialised = model[0].get_model()

    elif len(model) > 1:
        model_initialised = select_model(
            train=train,
            target_column=target_column,
            weight_column=weight_column,
            models_to_test=model,
            output_folder=output,
            classification_prediction=classification_prediction,
        )
    else:
        LOG.error(
            "Model incorrectly provided or not provided at all \
                          Provide a valid model(s) from the Models Enum class."
        )
        raise ValueError(
            "Model incorrectly provided or not provided at all \
                          Provide a valid model(s) from the Models Enum class."
        )

    (model_fit, residuals, x_train, x_test, mse) = initialise_model(
        train=train,
        target_column=target_column,
        output_folder=output,
        weight_column=weight_column,
        model_initialised=model_initialised,
        classification_prediction=classification_prediction,
    )

    return model_initialised, model_fit, residuals, x_test, x_train, mse

# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from pathlib import Path
import pandas as pd
from caf.brain.ml.model_selection.model_selection_functions import select_model, initialise_model
import logging
LOG = logging.getLogger(__name__)

def main_model_selection(train: pd.DataFrame,
                         target_column: str,
                         weight_column: str,
                         output: Path,
                         model,
                         classification_prediction: tuple[int, ...]):
    """
    Main function for selecting model algorithm and finding relevant algorithm
    attributes (if applicable).

    :param train: processed input data split into train subset.
    :param target_column: String column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param output: Path to output location.
    :param model: List or one algorithm to use as the base of the model.
                  Available algorithms can be seen in prediction_model_inputs.py
                  or __info__.py.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.

    :return:
        model_initialised: Initialised model algorithm from Models enum class.
        model_fit: Model fit on training data.
        residuals: Truth - predictions (based on training data).
        x_test, x_train, y_train: Training data split through train_test_split
                                  SciKitLearn function.
        mse: Mean squared error or None. Dependency on if the algorithm selected
             has coefficient values.

    """

    if not isinstance(model, list):
        model = [model]

    if len(model) == 1:
        model_initialised = model[0].get_model()

    elif len(model) > 1:
        model_initialised = select_model(train=train,
                                         target_column=target_column,
                                         weight_column=weight_column,
                                         models_to_test=model,
                                         output_folder=output,
                                         classification_prediction=classification_prediction)
    else:
        LOG.error("Model incorrectly provided or not provided at all \
                          Provide a valid model(s) from the Models Enum class.")
        raise ValueError("Model incorrectly provided or not provided at all \
                          Provide a valid model(s) from the Models Enum class.")

    (model_fit, residuals,
     x_train, x_test,
     y_train, y_test,
     mse) = initialise_model(train=train,
                             target_column=target_column,
                             output_folder=output,
                             weight_column=weight_column,
                             model_initialised=model_initialised,
                             classification_prediction=classification_prediction)

    return model_initialised, model_fit, residuals, x_test, x_train, y_train, mse

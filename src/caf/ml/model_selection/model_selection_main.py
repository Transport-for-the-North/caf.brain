# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.model_selection.model_selection_functions import select_model, initialise_model
import statsmodels.api as sm

def main_model_selection(train,
                         target_column,
                         weight_column,
                         output,
                         model,
                         classification_prediction):

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
        raise ValueError("Model incorrectly provided or not provided at all \
                          Provide a valid model(s) from the Models Enum class.")

    (model_fit, residuals,
     x_train, x_test,
     y_train, y_test,
     mse) = initialise_model(train=train,
                             target_column=target_column,
                             output_folder=output,
                             weight_column=weight_column,
                             model_initialised=model_initialised)

    return model_initialised, model_fit, residuals, x_test, x_train, y_train, mse

# -*- coding: utf-8 -*-
"""
Created on: 1/24/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.prediction.prediction_functions import prediction


def main_prediction(model,
                    test,
                    target_column,
                    output_folder,
                    validation,
                    weight_column,
                    binary_prediction):

    y_pred = prediction(model=model,
                        test=test,
                        target_column=target_column,
                        output_folder=output_folder,
                        validation=validation,
                        weight_column=weight_column,
                        binary_prediction=binary_prediction)
    return y_pred

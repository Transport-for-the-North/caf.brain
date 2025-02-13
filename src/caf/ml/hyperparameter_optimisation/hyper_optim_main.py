# -*- coding: utf-8 -*-
"""
Created on: 1/21/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.hyperparameter_optimisation.hyper_optim_functions import select_param
from pathlib import Path
import pandas as pd

def main_hyperparameter_optimisation(train_final: pd.DataFrame,
                                     target_column: str,
                                     model_instance,
                                     model_name,
                                     classification_prediction: tuple[int, ...],
                                     cv: str,
                                     weight_column: str,
                                     output_folder: Path,
                                     is_time_series: bool):
    """
    Main function for hyperparameter optimisation.

    :param train_final: Dataframe of final training data post feature selection.
    :param target_column: String column name of value to predict.
    :param model_instance: Initialised model algorithm from Models enum class.
    :param model_name: List or one algorithm to use as the base of the model.
                       Available algorithms can be seen in prediction_model_inputs.py
                       or __info__.py.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.
    :param cv: Cross validation method passed as a string. Any popular
               SciKitlearn methods are suitable with KFold being default if
               left as None.
    :param weight_column: Optional string column value to be used as weight.
    :param output_folder: Path to output location.
    :param is_time_series: If true then data must be time series. Time series
                           based characteristics are taken into consideration
                           during function execution.

    :return:
        best_model: Fitted final model for prediction on unseen (test) data.
    """

    best_model = select_param(train_final=train_final,
                              target_column=target_column,
                              model_instance=model_instance,
                              model_name=model_name,
                              classification_prediction=classification_prediction,
                              cv=cv,
                              weight_column=weight_column,
                              output_folder=output_folder,
                              is_time_series=is_time_series)

    return best_model

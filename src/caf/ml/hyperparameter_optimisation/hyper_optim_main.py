# -*- coding: utf-8 -*-
"""
Created on: 1/21/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.hyperparameter_optimisation.hyper_optim_functions import select_param


def main_hyperparameter_optimisation(train_final,
                                     target_column,
                                     model_instance,
                                     model_name,
                                     binary_prediction,
                                     cv,
                                     weight_column,
                                     output_folder):

    best_model = select_param(train_final=train_final,
                              target_column=target_column,
                              model_instance=model_instance,
                              model_name=model_name,
                              binary_prediction=binary_prediction,
                              cv=cv,
                              weight_column=weight_column,
                              output_folder=output_folder)

    return best_model

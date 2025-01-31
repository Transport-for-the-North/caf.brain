# -*- coding: utf-8 -*-
"""
Created on: 1/17/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.data_analysis.data_analysis_functions import pre_forecast_data_analysis



def main_evaluate_input_data(model_fit,
                             model_initialised,
                             residuals,
                             x_test,
                             train_scaled,
                             test_scaled,
                             full_transformations,
                             train_unscaled,
                             test_unscaled,
                             numerical_features,
                             categorical_features,
                             target_column,
                             weight_column,
                             x_train,
                             output_folder,
                             is_time_series):

    train_transformed, test_transformed = pre_forecast_data_analysis(residuals=residuals,
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
                                                                     is_time_series=is_time_series)

    if train_transformed is not None:
        return train_transformed, test_transformed
    else:
        return train_scaled, test_scaled

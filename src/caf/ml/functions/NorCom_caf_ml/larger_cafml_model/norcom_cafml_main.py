# -*- coding: utf-8 -*-
"""
Created on: 10/9/2024
Original author: Adil Zaheer
"""
import os
import time

import pandas as pd
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.functions.NorCom_caf_ml.larger_cafml_model.norcom_saved_model_funcs import saved_model_func
from caf.ml.functions.NorCom_caf_ml.larger_cafml_model.norcom_specific_functions import (norcom_run_functions,
                                                                                         refined_data_processor_function,
                                                                                         encode_and_sort)
from caf.ml.functions.save_model import load_model_and_parameters
from caf.ml.functions.NorCom_caf_ml.larger_cafml_model.norcom_inputs import NorCom_inputs


def main(params: NorCom_inputs):
    saved_model_path = os.path.join(params.output_folder, 'regression_method.pkl')
    if os.path.exists(saved_model_path):
        return process_saved(params)
    else:
        return process_prediction(params)


def process_prediction(params):
    training_dat_path = os.path.join(params.output_folder, 'training_data.csv')
    if os.path.exists(training_dat_path):
        training = pd.read_csv(training_dat_path)
        test = pd.read_csv(os.path.join(params.output_folder, 'test_data.csv'))
        validation = pd.read_csv(os.path.join(params.output_folder, 'validation_data.csv'))

        training.set_index(params.index_columns, inplace=True)
        test.set_index(params.index_columns, inplace=True)
        validation.set_index(params.index_columns, inplace=True)

    else:
        processed_data, index_columns_df = refined_data_processor_function(df=params.classified_build,
                                                                           target_column=params.target_column,
                                                                           index_columns=params.index_columns,
                                                                           categorical_features=params.categorical_features,
                                                                           numerical_features=params.numerical_features,
                                                                           output_folder=params.output_folder,
                                                                           column_name_to_drop_rows=params.column_name_to_drop_rows,
                                                                           value_in_row=params.value_in_row,
                                                                           weight_column=params.weight_column)

        training, test, validation = encode_and_sort(df=processed_data,
                                                     target_column=params.target_column,
                                                     output_folder=params.output_folder,
                                                     categorical_feat=params.categorical_features,
                                                     training_year=params.training_year,
                                                     weight_column=params.weight_column,
                                                     binary_prediction=params.binary_prediction)

    return norcom_run_functions(params, training=training, test=test, validation=validation)


def process_saved(params):
    regression_method, hyperparameters, trained_data, transformations, data_analysis_data = load_model_and_parameters(
        params.output_folder)
    return saved_model_func(params, regression_method, hyperparameters, trained_data, transformations, data_analysis_data)

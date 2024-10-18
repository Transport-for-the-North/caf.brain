# -*- coding: utf-8 -*-
"""
Created on: 10/9/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.functions.data_pipeline_functions import process_data_pipeline
from caf.ml.functions.NorCom_caf_ml.norcom_saved_model_funcs import saved_model_func
from caf.ml.functions.NorCom_caf_ml.norcom_specific_functions import norcom_run_functions
from caf.ml.functions.process_data_class import DataProcessor
from caf.ml.functions.save_model import load_model_and_parameters
from caf.ml.inputs.cafml_inputs import NorCom_cafml_inputs


def main(params: NorCom_cafml_inputs):
    if params.saved_model:
        return process_saved_model(params)
    else:
        return process_new_model(params)


def process_new_model(params):
    return process_prediction(params)


def process_saved_model(params):
    return process_saved(params)


def process_prediction(params):
    processed_data = DataProcessor(x=params.x_path,
                                   y=None,
                                   folder_path=None,
                                   index_columns=params.index_columns,
                                   drop_columns=params.drop_columns,
                                   keep_columns=params.keep_columns,
                                   target_column=params.target_column,
                                   output_folder=params.output_folder,
                                   wide_format=None,
                                   value_name=None,
                                   variable_name=None,
                                   outlier_threshold=None,
                                   categorical_target=params.categorical_target,
                                   column_name_to_drop_rows=params.column_name_to_drop_rows,
                                   value_in_row=params.value_in_row).data

    preprocessed_df, transformations_ = process_data_pipeline(df=processed_data,
                                                              numerical_features=params.numerical_features,
                                                              categorical_features=params.categorical_features,
                                                              target_column=params.target_column,
                                                              output_folder=params.output_folder)

    return norcom_run_functions(params, preprocessed_df=preprocessed_df, transformations_=transformations_)


def process_saved(params):
    regression_method, hyperparameters, trained_data, transformations, data_analysis_data = load_model_and_parameters(
        params.output_folder)
    return saved_model_func(params, regression_method, hyperparameters, trained_data, transformations, data_analysis_data)

# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from src.caf.ml.CODE_OVERHAUL.inputs_and_baseclasses.run_inputs import run_file_inputs
from caf.ml.CODE_OVERHAUL.process_data_functions.process_data_main import main_input_data
#TODO NORCOM: make scratches inside repo that can be example runs / run outlines


def main(params: run_file_inputs):
    train, test, validate = main_input_data(output_path=params.output_path,
                                            file_path=params.file_path,
                                            folder_path=params.folder_path,
                                            target_column=params.target_column,
                                            custom_index=params.custom_index,
                                            column_name_to_drop_rows=params.column_name_to_drop_rows,
                                            value_in_row=params.value_in_row,
                                            weight_column=params.weight_column,
                                            categorical_features=params.categorical_features,
                                            numerical_features=params.numerical_features,
                                            binary_prediction=params.binary_prediction,
                                            time_series_split=params.time_series_split,
                                            validation_path=params.validation_path)



    return

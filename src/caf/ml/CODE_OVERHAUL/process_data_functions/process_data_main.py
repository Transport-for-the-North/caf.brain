# -*- coding: utf-8 -*-
"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import pandas as pd
import os as os
from caf.ml.CODE_OVERHAUL.process_data_functions.encode_and_scale import process_data_pipeline
from caf.ml.CODE_OVERHAUL.process_data_functions.split_data_into_ttv import split_data
from src.caf.ml.CODE_OVERHAUL.inputs_and_baseclasses.run_inputs import run_file_inputs
from caf.ml.CODE_OVERHAUL.process_data_functions.process_input_data_functions import InitialDataProcessing


#TODO NORCOM: make scratches inside repo that can be example runs / run outlines


def main_input_data(params: run_file_inputs):
    train = None
    test = None
    validate = None

    params.output_path = os.path.join(params.output_path, 'output')
    if not os.path.exists(params.output_path):
        os.makedirs(params.output_path)

    processor = InitialDataProcessing(file_path=params.file_path,
                                      folder_path=params.folder_path,
                                      output_path=params.output_path,
                                      target_column=params.target_column,
                                      custom_index=params.custom_index,
                                      column_name_to_drop_rows=params.column_name_to_drop_rows,
                                      value_in_row=params.value_in_row,
                                      weight_column=params.weight_column,
                                      categorical_features=params.categorical_features,
                                      numerical_features=params.numerical_features,
                                      binary_prediction=params.binary_prediction)

    processed_dataframes = processor.execute_pipeline()

    if len(processed_dataframes) == 1:
        df = list(processed_dataframes.values())[0]
    else:
        df = pd.concat(processed_dataframes.values(), axis=0)

    print(df.head(10))
    print(df.dtypes)
    processed_dataframes = process_data_pipeline(df=df,
                                                 numerical_features=params.numerical_features,
                                                 categorical_features=params.categorical_features,
                                                 target_column=params.target_column)


    train, test, validate = split_data(processed_dataframes=processed_dataframes,
                                       index_columns=params.custom_index,
                                       weight_column=params.weight_column,
                                       target_column=params.target_column,
                                       time_series_split=params.time_series_split,
                                       validation_path=params.validation_path,
                                       output_path=params.output_path)

    return train, test, validate

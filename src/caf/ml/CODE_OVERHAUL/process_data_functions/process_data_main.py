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
from caf.ml.CODE_OVERHAUL.process_data_functions.process_input_data_functions import InitialDataProcessing


def main_input_data(output_path,
                    file_path,
                    folder_path,
                    target_column,
                    custom_index,
                    column_name_to_drop_rows,
                    value_in_row,
                    weight_column,
                    categorical_features,
                    numerical_features,
                    binary_prediction,
                    time_series_split,
                    validation_path):


    output_path = os.path.join(output_path, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    processor = InitialDataProcessing(file_path=file_path,
                                      folder_path=folder_path,
                                      output_path=output_path,
                                      target_column=target_column,
                                      custom_index=custom_index,
                                      column_name_to_drop_rows=column_name_to_drop_rows,
                                      value_in_row=value_in_row,
                                      weight_column=weight_column,
                                      categorical_features=categorical_features,
                                      numerical_features=numerical_features,
                                      binary_prediction=binary_prediction)

    processed_dataframes = processor.execute_pipeline()

    if len(processed_dataframes) == 1:
        df = list(processed_dataframes.values())[0]
    else:
        df = pd.concat(processed_dataframes.values(), axis=0)

    processed_dataframes = process_data_pipeline(df=df,
                                                 numerical_features=numerical_features,
                                                 categorical_features=categorical_features,
                                                 target_column=target_column)


    train, test, validate = split_data(processed_dataframes=processed_dataframes,
                                       index_columns=custom_index,
                                       weight_column=weight_column,
                                       target_column=target_column,
                                       time_series_split=time_series_split,
                                       validation_path=validation_path,
                                       output_path=output_path)

    return train, test, validate

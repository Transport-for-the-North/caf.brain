# -*- coding: utf-8 -*-
"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import pandas as pd
import os as os
from caf.ml.process_data_functions.encode_and_scale import process_data_pipeline
from caf.ml.process_data_functions.split_data_into_ttv import split_data
from caf.ml.process_data_functions.process_input_data_functions import InitialDataProcessing

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
                    validation_path,
                    split_size):


    if os.path.exists(os.path.join(output_path, 'train.csv')):
        train_raw = pd.read_csv(os.path.join(output_path, 'train.csv'))
        test_raw = pd.read_csv(os.path.join(output_path, 'test.csv'))
        try:
            validate = pd.read_csv(os.path.join(output_path, 'validate.csv'))
            validate[target_column] = validate[target_column].astype(float)

        except FileNotFoundError:
            print("Validate not provided. Validation will not be \
                   preformed")
            validate = None

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


        processed_dfs = {}
        for name, df in [('train', train_raw), ('test', test_raw)]:
            processor.df = df.copy()
            processor.dataframes = {}

            current_name = name
            is_test_data = current_name.lower() == 'test'

            processed = processor.data_already_split_pipeline(is_test_data)

            processed_df = list(processed.values())[0]
            processed_dfs[name] = processed_df

            print(f"Processed {name} dataframe:")
            print(f"Index names: {processed_df.index.names}")
            print(f"Columns: {processed_df.columns.tolist()}")
            print(f"Shape: {processed_df.shape}")
            print("---")

        train_unscaled = processed_dfs['train']
        test_unscaled = processed_dfs['test']
        train_unscaled[target_column] = train_unscaled[target_column].astype(int)

        train_scaled = process_data_pipeline(df=train_unscaled.copy(),
                                             numerical_features=numerical_features,
                                             categorical_features=categorical_features,
                                             target_column=target_column)

        test_scaled = process_data_pipeline(df=test_unscaled.copy(),
                                            numerical_features=numerical_features,
                                            categorical_features=categorical_features,
                                            target_column=target_column)

        data_dict = {
            'train_scaled': train_scaled,
            'test_scaled': test_scaled,
            'train_unscaled': train_unscaled,
            'test_unscaled': test_unscaled,
            'validate': validate
        }
        return data_dict

    else:

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

        is_test_data = False
        processed_dataframes = processor.execute_pipeline(is_test_data)
        if len(processed_dataframes) == 1:
            df = list(processed_dataframes.values())[0]
        else:
            df = pd.concat(processed_dataframes.values(), axis=0)

        train_unscaled, test_unscaled, validate = split_data(processed_dataframes=df,
                                                             index_columns=custom_index,
                                                             weight_column=weight_column,
                                                             target_column=target_column,
                                                             time_series_split=time_series_split,
                                                             validation_path=validation_path,
                                                             output_path=output_path,
                                                             split_size=split_size)

        train_scaled = process_data_pipeline(df=train_unscaled.copy(),
                                             numerical_features=numerical_features,
                                             categorical_features=categorical_features,
                                             target_column=target_column)

        test_scaled = process_data_pipeline(df=test_unscaled.copy(),
                                            numerical_features=numerical_features,
                                            categorical_features=categorical_features,
                                            target_column=target_column)

        train_unscaled[target_column] = train_unscaled[target_column].astype(int)
        train_scaled[target_column] = train_scaled[target_column].astype(int)

        if validate:
            validate[target_column] = validate[target_column].astype(int)

        data_dict = {
            'train_scaled': train_scaled,
            'test_scaled': test_scaled,
            'train_unscaled': train_unscaled,
            'test_unscaled': test_unscaled,
            'validate': validate
        }

        return data_dict

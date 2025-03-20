# -*- coding: utf-8 -*-
"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import pandas as pd
import os as os
from pathlib import Path
from typing import List
from caf.brain.ml.process_data_functions.encode_and_scale import process_data_pipeline
from caf.brain.ml.process_data_functions.split_data_into_ttv import split_data
from caf.brain.ml.process_data_functions.process_input_data_functions import InitialDataProcessing
import logging
LOG = logging.getLogger(__name__)

def main_input_data(output_path: Path,
                    file_path: Path,
                    folder_path: Path,
                    target_column: str,
                    custom_index: List[str],
                    column_name_to_drop_rows: List[str],
                    value_in_row: List[str],
                    weight_column: str,
                    categorical_features: List[str],
                    numerical_features: List[str],
                    classification_prediction: tuple[int, ...],
                    split_by_value: str,
                    validation_path: Path,
                    split_size: float,
                    sample_size_encode: bool,
                    select_encode_values: bool,
                    encode_values_to_drop: List[str]) -> dict:
    """
    Main function for processing input data.

    :param output_path: Path to output location.
    :param file_path: Optional path to data to be used for modelling.
    :param folder_path: Optional path to folder of data to be used for modelling.
    :param target_column: String column name of value to predict.
    :param custom_index: list of string column names to be used as an index.
                         Must be one value e.g. year if splitting data
                         into train and test via this column. Corresponds to
                         split_by_value in this case.
    :param column_name_to_drop_rows: list of string column names that
                                     contain values to drop.
    :param value_in_row: corresponding values for column_name_to_drop_rows.
    :param weight_column: Optional string column value to be used as weight.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :param numerical_features: List of string column names that are
                               continuous variables.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.
    :param split_by_value: Optional string that links to custom_index. The
                           value in the index column to split the data into
                           training and test.
    :param validation_path: Optional path to validation data if it exists.
                            This would need to correspond to the test data
                            created.
    :param split_size: Optional float e.g. 0.2. This would be the ratio to
                       randomly split data into train and test. 0.2 is used
                       if left as None.
    :param sample_size_encode: Optional bool. If true, the data will be split
                               based on sample size. Variables with the largest
                               sample size will be used as reference class.
    :param select_encode_values: Optional bool. If True, data is split based
                                 on custom values set by the user. Corresponds
                                 to encode_values_to_drop.
    :param encode_values_to_drop: If select_encode_values is True, then this
                                  must be a list of strings the length of
                                  categorical_features. Position one in the list
                                  will link to the first variable provided in
                                  categorical_features and so on.

    :return:
        Dictionary of processed dataframes.
        drop_vals: Values dropped during encoding of categorical variables.
    """

    folder_path = "" if folder_path is None else folder_path

    if (os.path.exists(os.path.join(output_path, 'train.csv')) or
            os.path.exists(os.path.join(folder_path, 'train.csv'))):
        # try output_path
        try:
            train_raw = pd.read_csv(os.path.join(output_path, 'train.csv'))
            test_raw = pd.read_csv(os.path.join(output_path, 'test.csv'))
            try:
                validate = pd.read_csv(os.path.join(output_path, 'validate.csv'))
                validate[target_column] = validate[target_column].astype(float)
            except FileNotFoundError:
                LOG.warning("Validate not provided. Validation will not be performed")
                validate = None
        # try folder_path
        except FileNotFoundError:
            train_raw = pd.read_csv(os.path.join(folder_path, 'train.csv'))
            test_raw = pd.read_csv(os.path.join(folder_path, 'test.csv'))
            try:
                validate = pd.read_csv(os.path.join(folder_path, 'validate.csv'))
                validate[target_column] = validate[target_column].astype(float)
            except FileNotFoundError:
                LOG.warning("Validate not provided. Validation will not be performed")
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
                                          classification_prediction=classification_prediction)


        processed_dfs = {}
        for name, df in [('train', train_raw), ('test', test_raw)]:
            processor.df = df.copy()
            processor.dataframes = {}

            current_name = name
            is_test_data = current_name.lower() == 'test'

            processed = processor.data_already_split_pipeline(is_test_data)

            processed_df = list(processed.values())[0]
            processed_dfs[name] = processed_df

            LOG.info(f"Processed {name} dataframe:")
            LOG.info(f"Index names: {processed_df.index.names}")
            LOG.info(f"Columns: {processed_df.columns.tolist()}")
            LOG.info(f"Shape: {processed_df.shape}")

        train_unscaled = processed_dfs['train']
        test_unscaled = processed_dfs['test']
        train_unscaled[target_column] = train_unscaled[target_column].astype(int)

        train_scaled, drop_vals, numerical_pipeline = process_data_pipeline(df=train_unscaled.copy(),
                                                                            numerical_features=numerical_features,
                                                                            categorical_features=categorical_features,
                                                                            target_column=target_column,
                                                                            weight_column=weight_column,
                                                                            sample_size_encode=sample_size_encode,
                                                                            select_encode_values=select_encode_values,
                                                                            encode_values_to_drop=encode_values_to_drop,
                                                                            train_encoded=None,
                                                                            test_data=False,
                                                                            numerical_pipeline=None,
                                                                            output_folder=output_path)

        test_scaled, _, _ = process_data_pipeline(df=test_unscaled.copy(),
                                                  numerical_features=numerical_features,
                                                  categorical_features=categorical_features,
                                                  target_column=target_column,
                                                  weight_column=weight_column,
                                                  sample_size_encode=sample_size_encode,
                                                  select_encode_values=select_encode_values,
                                                  encode_values_to_drop=encode_values_to_drop,
                                                  train_encoded=train_scaled,
                                                  test_data=True,
                                                  numerical_pipeline=numerical_pipeline,
                                                  output_folder=output_path)

        data_dict = {
            'train_scaled': train_scaled,
            'test_scaled': test_scaled,
            'train_unscaled': train_unscaled,
            'test_unscaled': test_unscaled,
            'validate': validate
        }
        return data_dict, drop_vals, numerical_pipeline

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
                                          classification_prediction=classification_prediction)

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
                                                             split_by_value=split_by_value,
                                                             validation_path=validation_path,
                                                             output_path=output_path,
                                                             split_size=split_size,
                                                             categorical_features=categorical_features)

        if validate is not None:
            validate[target_column] = validate[target_column].astype(int)

        train_scaled, drop_vals, numerical_pipeline = process_data_pipeline(df=train_unscaled.copy(),
                                                                            numerical_features=numerical_features,
                                                                            categorical_features=categorical_features,
                                                                            target_column=target_column,
                                                                            weight_column=weight_column,
                                                                            sample_size_encode=sample_size_encode,
                                                                            select_encode_values=select_encode_values,
                                                                            encode_values_to_drop=encode_values_to_drop,
                                                                            train_encoded=None,
                                                                            test_data=False,
                                                                            numerical_pipeline=None,
                                                                            output_folder=output_path)

        test_scaled, _, _ = process_data_pipeline(df=test_unscaled.copy(),
                                                  numerical_features=numerical_features,
                                                  categorical_features=categorical_features,
                                                  target_column=target_column,
                                                  weight_column=weight_column,
                                                  sample_size_encode=sample_size_encode,
                                                  select_encode_values=select_encode_values,
                                                  encode_values_to_drop=encode_values_to_drop,
                                                  train_encoded=train_scaled,
                                                  test_data=True,
                                                  numerical_pipeline=numerical_pipeline,
                                                  output_folder=output_path)

        train_unscaled[target_column] = train_unscaled[target_column].astype(int)
        train_scaled[target_column] = train_scaled[target_column].astype(int)


        data_dict = {
            'train_scaled': train_scaled,
            'test_scaled': test_scaled,
            'train_unscaled': train_unscaled,
            'test_unscaled': test_unscaled,
            'validate': validate
        }

        return data_dict, drop_vals, numerical_pipeline

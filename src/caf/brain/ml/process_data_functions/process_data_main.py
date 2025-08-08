# -*- coding: utf-8 -*-
"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""
# Built-Ins
import logging
import os as os
from pathlib import Path
from typing import List

# Third Party
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import pandas as pd

# Local Imports
from caf.brain.ml.process_data_functions.encode_and_scale import process_data_pipeline
from caf.brain.ml.process_data_functions.process_input_data_functions import (
    InitialDataProcessing,
)
from caf.brain.ml.process_data_functions.split_data_into_ttv import split_data

LOG = logging.getLogger(__name__)


def main_input_data(
    output_path: Path,
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
    encode_values_to_drop: List[str],
) -> dict:
    """
    Main function for processing input data.

    Parameters
    ----------
    output_path : pathlib.Path
        Path to output location.
    file_path : pathlib.Path
        Optional path to data to be used for modelling.
    folder_path : pathlib.Path
        Optional path to folder of data to be used for modelling.
    target_column : str
        Name of the column to predict.
    custom_index : list of str
        List of column names to be used as an index. Must be one value (e.g. year)
        if splitting data into train and test via this column. Corresponds to
        split_by_value in this case.
    column_name_to_drop_rows : list of str
        List of column names that contain values to drop.
    value_in_row : list of str
        Corresponding values for column_name_to_drop_rows.
    weight_column : str
        Optional column name to be used as sample weights.
    categorical_features : list of str
        List of column names that are categorical variables.
    numerical_features : list of str
        List of column names that are continuous variables.
    classification_prediction : tuple of int
        Target values to predict in a classification problem.
    split_by_value : str
        Optional string that links to custom_index. The value in the index column
        to split the data into training and test.
    validation_path : pathlib.Path
        Optional path to validation data if it exists. This should correspond to
        the test data created.
    split_size : float
        Ratio to randomly split data into train and test (e.g. 0.2). 0.2 is used
        if left as None.
    sample_size_encode : bool
        If True, the data will be split based on sample size. Variables with the
        largest sample size will be used as reference class.
    select_encode_values : bool
        If True, data is split based on custom values set by the user. Corresponds
        to encode_values_to_drop.
    encode_values_to_drop : list of str
        If select_encode_values is True, then this must be a list of strings the
        length of categorical_features. Position one in the list will link to the
        first variable provided in categorical_features and so on.

    Returns
    -------
    data_dict : dict
        Dictionary of processed dataframes with keys:
        'train_scaled', 'test_scaled', 'train_unscaled', 'test_unscaled', 'validate'.
    drop_vals : pandas.DataFrame or None
        Values dropped during encoding of categorical variables.
    numerical_pipeline : sklearn.Pipeline or None
        Fitted pipeline for numerical features.
    """

    folder_path = "" if folder_path is None else folder_path

    if os.path.exists(os.path.join(output_path, "train.csv")) or os.path.exists(
        os.path.join(folder_path, "train.csv")
    ):
        # try output_path
        try:
            train_raw = pd.read_csv(os.path.join(output_path, "train.csv"))
            test_raw = pd.read_csv(os.path.join(output_path, "test.csv"))
            try:
                validate = pd.read_csv(os.path.join(output_path, "validate.csv"))
                validate[target_column] = validate[target_column].astype(float)
            except FileNotFoundError:
                LOG.warning("Validate not provided. Validation will not be performed")
                validate = None
        # try folder_path
        except FileNotFoundError:
            train_raw = pd.read_csv(os.path.join(folder_path, "train.csv"))
            test_raw = pd.read_csv(os.path.join(folder_path, "test.csv"))
            try:
                validate = pd.read_csv(os.path.join(folder_path, "validate.csv"))
                validate[target_column] = validate[target_column].astype(float)
            except FileNotFoundError:
                LOG.warning("Validate not provided. Validation will not be performed")
                validate = None

        processor = InitialDataProcessing(
            file_path=file_path,
            folder_path=folder_path,
            output_path=output_path,
            target_column=target_column,
            custom_index=custom_index,
            column_name_to_drop_rows=column_name_to_drop_rows,
            value_in_row=value_in_row,
            weight_column=weight_column,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            classification_prediction=classification_prediction,
        )

        processed_dfs = {}
        for name, df in [("train", train_raw), ("test", test_raw)]:
            processor.df = df.copy()
            processor.dataframes = {}

            current_name = name
            is_test_data = current_name.lower() == "test"

            processed = processor.data_already_split_pipeline(is_test_data)

            processed_df = list(processed.values())[0]
            processed_dfs[name] = processed_df

            LOG.info("Processed %s dataframe:", name)
            LOG.info("Index names: %s", processed_df.index.names)
            LOG.info("Columns: %s", processed_df.columns.tolist())
            LOG.info("Shape: %s", processed_df.shape)

        train_unscaled = processed_dfs["train"]
        test_unscaled = processed_dfs["test"]
        train_unscaled[target_column] = train_unscaled[target_column].astype(int)

        train_scaled, drop_vals, numerical_pipeline = process_data_pipeline(
            df=train_unscaled.copy(),
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
            output_folder=output_path,
        )

        test_scaled, _, _ = process_data_pipeline(
            df=test_unscaled.copy(),
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
            output_folder=output_path,
        )

        data_dict = {
            "train_scaled": train_scaled,
            "test_scaled": test_scaled,
            "train_unscaled": train_unscaled,
            "test_unscaled": test_unscaled,
            "validate": validate,
        }
        return data_dict, drop_vals, numerical_pipeline

    else:
        processor = InitialDataProcessing(
            file_path=file_path,
            folder_path=folder_path,
            output_path=output_path,
            target_column=target_column,
            custom_index=custom_index,
            column_name_to_drop_rows=column_name_to_drop_rows,
            value_in_row=value_in_row,
            weight_column=weight_column,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            classification_prediction=classification_prediction,
        )

        is_test_data = False
        processed_dataframes = processor.execute_pipeline(is_test_data)
        if len(processed_dataframes) == 1:
            df = list(processed_dataframes.values())[0]
        else:
            df = pd.concat(processed_dataframes.values(), axis=0)

        train_unscaled, test_unscaled, validate = split_data(
            processed_dataframes=df,
            index_columns=custom_index,
            weight_column=weight_column,
            target_column=target_column,
            split_by_value=split_by_value,
            validation_path=validation_path,
            output_path=output_path,
            split_size=split_size,
            categorical_features=categorical_features,
        )

        if validate is not None:
            validate[target_column] = validate[target_column].astype(int)

        train_scaled, drop_vals, numerical_pipeline = process_data_pipeline(
            df=train_unscaled.copy(),
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
            output_folder=output_path,
        )

        test_scaled, _, _ = process_data_pipeline(
            df=test_unscaled.copy(),
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
            output_folder=output_path,
        )

        train_unscaled[target_column] = train_unscaled[target_column].astype(int)
        train_scaled[target_column] = train_scaled[target_column].astype(int)

        data_dict = {
            "train_scaled": train_scaled,
            "test_scaled": test_scaled,
            "train_unscaled": train_unscaled,
            "test_unscaled": test_unscaled,
            "validate": validate,
        }

        return data_dict, drop_vals, numerical_pipeline

# -*- coding: utf-8 -*-
"""
Created on: 1/15/2025
Original author: Adil Zaheer
"""
import os.path
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from pathlib import Path
import pandas as pd
from caf.ml.CODE_OVERHAUL.process_data_functions.process_input_data_functions import InitialDataProcessing
from sklearn.model_selection import train_test_split


def split_data(processed_dataframes: dict,
               index_columns: list[str],
               weight_column: str,
               time_series_split: str,
               validation_path: Path,
               output_path: Path):
    """

    :param processed_dataframes:
    :param index_columns:
    :param weight_column:
    :param time_series_split:
    :param validation_path:
    :param output_path:
    :return:
    """
    if isinstance(processed_dataframes, dict):
        df = pd.DataFrame.from_dict(processed_dataframes)
    else:
        df = processed_dataframes

    if time_series_split is not None:
        train = df.loc[df.index.get_level_values(index_columns) <= int(time_series_split)]
        test = df.loc[df.index.get_level_values(index_columns) > int(time_series_split)]
        if weight_column in test:
            test = test.drop(columns=weight_column)

    else:
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        if weight_column in test:
            test = test.drop(columns=weight_column)

    validate = InitialDataProcessing.read_file(file_path=validation_path)
    if index_columns in validate.columns:
        validate = validate.set_index(index_columns)

    train.to_csv(os.path.join(output_path, 'train.csv'), index=True)
    test.to_csv(os.path.join(output_path, 'test.csv'), index=True)

    return train, test, validate

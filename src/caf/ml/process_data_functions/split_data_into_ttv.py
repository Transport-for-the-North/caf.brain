# -*- coding: utf-8 -*-
"""
Created on: 1/15/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os.path
from pathlib import Path
import pandas as pd
from caf.ml.process_data_functions.process_input_data_functions import InitialDataProcessing
from sklearn.model_selection import train_test_split


def split_data(processed_dataframes: dict,
               index_columns: list[str],
               weight_column: str,
               time_series_split: str,
               target_column: str,
               validation_path: Path,
               output_path: Path,
               split_size: int):
    """

    :param split_size:
    :param target_column:
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
        if target_column in test:
            test = test.drop(columns=target_column)


    else:
        strat = pd.cut(df.iloc[:, 0], 4)
        train, test = train_test_split(df, test_size=split_size, random_state=42, stratify=strat)
        if weight_column in test:
            test = test.drop(columns=weight_column)
        if target_column in test:
            test = test.drop(columns=target_column)


    if validation_path is not None:
        validate = InitialDataProcessing.read_file(file_path=validation_path)
        if index_columns in validate.columns:
            validate = validate.set_index(index_columns)

        train.to_csv(os.path.join(output_path, 'train.csv'), index=True)
        test.to_csv(os.path.join(output_path, 'test.csv'), index=True)

        return train, test, validate

    else:
        train.to_csv(os.path.join(output_path, 'train.csv'), index=True)
        test.to_csv(os.path.join(output_path, 'test.csv'), index=True)

        return train, test, None

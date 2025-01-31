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
               split_by_value: str,
               target_column: str,
               validation_path: Path,
               output_path: Path,
               split_size: int,
               categorical_features: list[str]):

    if isinstance(processed_dataframes, dict):
        df = pd.DataFrame.from_dict(processed_dataframes)
    else:
        df = processed_dataframes

    if split_by_value is not None:

        train, test, validate = split_by_column_value(df=df,
                                                      index_columns=index_columns,
                                                      split_by_value=split_by_value,
                                                      weight_column=weight_column,
                                                      target_column=target_column,
                                                      validation_path=validation_path,
                                                      output_path=output_path)

        return train, test, validate

    else:
        train, test, validate = stratified_split_with_categories(df=df,
                                                                 categorical_features=categorical_features,
                                                                 target_column=target_column,
                                                                 weight_column=weight_column,
                                                                 split_size=split_size,
                                                                 validation_path=validation_path,
                                                                 index_columns=index_columns,
                                                                 output_path=output_path)
        return train, test, validate


def stratified_split_with_categories(df,
                                     categorical_features,
                                     target_column,
                                     weight_column,
                                     split_size,
                                     validation_path,
                                     index_columns,
                                     output_path):


    strat = pd.cut(df.iloc[:, 0], 4)
    train, test = train_test_split(df,
                                   test_size=split_size,
                                   random_state=42,
                                   stratify=strat)

    missing_categories = {}
    for col in categorical_features:
        unique_vals = set(df[col].unique())
        test_vals = set(test[col].unique())
        missing = unique_vals - test_vals
        if missing:
            missing_categories[col] = missing

    if missing_categories:
        for col, missing_vals in missing_categories.items():
            for val in missing_vals:
                missing_row = df[df[col] == val].iloc[[0]]
                test = pd.concat([test, missing_row])
                train = train[~train.index.isin(missing_row.index)]

    validate = None
    if weight_column in test.columns:
        test = test.drop(columns=weight_column)

    if target_column in test.columns:
        validate = pd.DataFrame({target_column: test[target_column]}, index=test.index)
        validate.to_csv(os.path.join(output_path, 'validate.csv'), index=True)
        test = test.drop(columns=target_column)

    if validation_path is not None:
        validate = InitialDataProcessing.read_file(file_path=validation_path)
        if index_columns in validate.columns:
            validate = validate.set_index(index_columns)

    train.to_csv(os.path.join(output_path, 'train.csv'), index=True)
    test.to_csv(os.path.join(output_path, 'test.csv'), index=True)
    return train, test, validate


def split_by_column_value(df,
                          index_columns,
                          split_by_value,
                          weight_column,
                          target_column,
                          validation_path,
                          output_path):
    train = df.loc[df.index.get_level_values(index_columns) <= int(split_by_value)]
    test = df.loc[df.index.get_level_values(index_columns) > int(split_by_value)]

    validate = None
    if weight_column in test:
        test = test.drop(columns=weight_column)
    if target_column in test:
        validate = pd.DataFrame({target_column: test[target_column]}, index=test.index)
        validate.to_csv(os.path.join(output_path, 'validate.csv'), index=True)
        test = test.drop(columns=target_column)

    train.to_csv(os.path.join(output_path, 'train.csv'), index=True)
    test.to_csv(os.path.join(output_path, 'test.csv'), index=True)

    if validation_path is not None:
        validate = InitialDataProcessing.read_file(file_path=validation_path)
        if index_columns in validate.columns:
            validate = validate.set_index(index_columns)

    return train, test, validate

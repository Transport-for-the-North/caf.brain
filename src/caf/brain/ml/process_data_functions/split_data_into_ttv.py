# -*- coding: utf-8 -*-
"""
Created on: 1/15/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os.path
from pathlib import Path
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from caf.brain.ml.main_models.prediction_model.prediction_model_inputs import PredictionModelInputs
from caf.brain.ml.process_data_functions.process_input_data_functions import (
    InitialDataProcessing,
)


def split_data(
    processed_dataframes: dict,
    paths: PredictionModelInputs.Paths,
    data_classification: PredictionModelInputs.DataClassificationInputs,
    transforming_inputs: PredictionModelInputs.TransformingInputDataInputs,
    output_path
) -> pd.DataFrame:
    """
    Function to split data into training and test if not already done by the
    user.

    Parameters
    ----------
    processed_dataframes: input dataframes inside a dictionary
    paths
    data_classification
    transforming_inputs
    output_path
    Returns
    -------
    train, test and validate dataframes.

    """
    if isinstance(processed_dataframes, dict):
        df = pd.DataFrame.from_dict(processed_dataframes)
    else:
        df = processed_dataframes

    if transforming_inputs.split_by_value is not None:

        train, test, validate = split_by_column_value(
            df=df,
            index_columns=data_classification.custom_index,
            split_by_value=transforming_inputs.split_by_value,
            weight_column=data_classification.weight_column,
            target_column=data_classification.target_column,
            validation_path=paths.validation_path,
            output_path=output_path,
        )

        return train, test, validate

    else:
        train, test, validate = stratified_split_with_categories(
            df=df,
            categorical_features=data_classification.categorical_features,
            target_column=data_classification.target_column,
            weight_column=data_classification.weight_column,
            split_size=transforming_inputs.split_size,
            validation_path=paths.validation_path,
            index_columns=data_classification.custom_index,
            output_path=output_path,
        )
        return train, test, validate


def stratified_split_with_categories(
    df: pd.DataFrame,
    categorical_features: List[str],
    target_column: str,
    weight_column: str,
    split_size: float,
    validation_path: Path,
    index_columns: List[str],
    output_path: Path,
) -> pd.DataFrame:
    """
    Function to split data into train, test and validate by using a value
    provided by the user. The value is the ratio in which to split the data
    randomly into train and test. The function ensures that all categories are
    represented at least once in both train and test. If split_size is None
    then 0.2 is used by default.

    :param df: input dataframe.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :param target_column: sting column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param split_size: Optional float e.g. 0.2. This would be the ratio to
                       randomly split data into train and test. 0.2 is used
                       if left as None.
    :param validation_path: Optional path to validation data if it exists.
                            This would need to correspond to the test data
                            created.
    :param index_columns: list of string column names to be used as an index.
    :param output_path: Path to output location.
    :return: Train, test and validate dataframes.
    """

    strat = pd.cut(df.iloc[:, 0], 4)
    train, test = train_test_split(
        df, test_size=split_size if not None else 0.2, random_state=42, stratify=strat
    )

    if categorical_features is not None:
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
        validate.to_csv(os.path.join(output_path, "validate.csv"), index=True)
        test = test.drop(columns=target_column)

    if validation_path is not None:
        validate = InitialDataProcessing.read_file(file_path=validation_path)
        if index_columns in validate.columns:
            validate = validate.set_index(index_columns)

    train.to_csv(os.path.join(output_path, "train.csv"), index=True)
    test.to_csv(os.path.join(output_path, "test.csv"), index=True)
    return train, test, validate


def split_by_column_value(
    df: pd.DataFrame,
    index_columns: List[str],
    split_by_value: str,
    weight_column: str,
    target_column: str,
    validation_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Function to split data into train, test and validate by using a value
    provided by the user. The value must correspond to the index column
    used. The index column must only be one column specified, multi-index is
    not applicable. E.g. index_columns: year, split_by_value '2019'. This would
    mean everything pre-2019 is training and everything post 2019 is test.

    :param df: input dataframe.
    :param index_columns: list of string column names to be used as an index.
                          Must be one value e.g. year if splitting data
                          into train and test via this column. Corresponds to
                          split_by_value in this case.
    :param split_by_value: Optional string that links to custom_index. The
                           value in the index column to split the data into
                           training and test.
    :param weight_column: Optional string column value to be used as weight.
    :param target_column: sting column name of value to predict.
    :param validation_path: Optional path to validation data if it exists.
                            This would need to correspond to the test data
                            created.
    :param output_path: Path to output location.
    :return: train, test and validate dataframes.
    """
    if not index_columns or len(index_columns) != 1:
        raise ValueError("index_columns must contain exactly one column name for splitting")

    index_column = index_columns[0]

    if index_column not in df.index.names and index_column != df.index.name:
        raise KeyError(f"Index column '{index_column}' not found in DataFrame index")

    train = df.loc[df.index.get_level_values(index_column) <= int(split_by_value)]
    test = df.loc[df.index.get_level_values(index_column) > int(split_by_value)]

    validate = None
    if weight_column in test:
        test = test.drop(columns=weight_column)
    if target_column in test:
        validate = pd.DataFrame({target_column: test[target_column]}, index=test.index)
        validate.to_csv(os.path.join(output_path, "validate.csv"), index=True)
        test = test.drop(columns=target_column)

    train.to_csv(os.path.join(output_path, "train.csv"), index=True)
    test.to_csv(os.path.join(output_path, "test.csv"), index=True)

    if validation_path is not None:
        validate = InitialDataProcessing.read_file(file_path=validation_path)
        if index_column in validate.columns:
            validate = validate.set_index(index_column)

    return train, test, validate


def simple_train_test_split(df: pd.DataFrame,
                            target_column: str,
                            weight_column: str):
    """
    Split data into train test split for model building purposes

    Parameters
    ----------
    df: Input pandas dataframe
    target_column: Y (dependent) variable in the dataframe (column)
    weight_column: Weight column in the dataframe

    Returns
    -------
    Dataframes containing the input data split into train and test as well as a
    Numpy ndarray of weight values.
    """
    x = df.drop(columns=[target_column])
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

    x_train_weight = None
    if weight_column in df.columns:
        x_train_weight = x_train[weight_column]
        x_train = x_train.drop(columns=weight_column)
        x_test = x_test.drop(columns=weight_column)

    return x_train, x_test, y_train, y_test, x_train_weight

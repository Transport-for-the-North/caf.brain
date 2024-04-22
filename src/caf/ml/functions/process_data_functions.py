# -*- coding: utf-8 -*-
"""
Created on: 16/11/2023
Original author: Adil Zaheer
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore


######### READ DATA FUNCTIONS #########

def read_folder(folder_path):
    dataframes_in = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            df = pd.read_csv(file_path, low_memory=False)
            dataframes_in[file_name] = df
    return dataframes_in


def read_csvs(x: pd.DataFrame, y: pd.DataFrame):
    x_in = pd.read_csv(x, low_memory=False)
    y_in = pd.read_csv(y, low_memory=False)
    return x_in, y_in


######### NUMERIC CONVERSION #########


def find_numeric_target_column(data: pd.DataFrame, target_column):
    # todo fix this to work without target column
    if target_column not in data.columns:
        return data

    # Check if the target column is present in the data
    if target_column in data.columns:
        # Check if the target column is numeric
        if not pd.to_numeric(data[target_column], errors='coerce').notna().all():
            data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
            if not data[target_column].notna().all():
                raise ValueError(
                    "The target column could not be converted to numeric.")
            print(
                f"The target column '{target_column}' has been converted to numeric.")
        return data

    raise ValueError(
        "Target column not found in the data. Modeling may require a numeric target column.")


def process_data_numeric(data, keep_columns=None, target_column=None, output_folder=None):
    if not isinstance(data, pd.DataFrame):
        data = convert_to_dataframe(data)

    data = data.apply(pd.to_numeric, errors='coerce')

    non_numeric_columns = data.columns[~data.applymap(np.isreal).all()]

    keep_columns = keep_columns or set()

    columns_to_drop = non_numeric_columns.difference(keep_columns)

    if target_column and target_column in columns_to_drop:
        raise ValueError(
            f"Target column '{target_column}' is still not numeric. Please review the data.")

    if not columns_to_drop.empty:
        print(f"Dropping non-numeric columns: {', '.join(columns_to_drop)}")
        data = data.drop(columns=columns_to_drop)

    if output_folder:
        output_file_path = os.path.join(output_folder, "non_numeric.csv")
        non_numeric_df = data[non_numeric_columns]
        non_numeric_df.to_csv(output_file_path, index=False)
        print(f"Non-numeric columns exported to: {output_file_path}")

    return data


######### SET DATAFRAME STRUCTURE #########


def index_sorter(*dataframes: pd.DataFrame, index_columns=None, drop_columns=None):

    if not dataframes:
        raise ValueError("At least one dataframe must be provided")

    def set_index(df: pd.DataFrame):
        if index_columns:
            return df.set_index(index_columns)
        else:
            return df

    result = [set_index(df) for df in dataframes]

    if drop_columns:
        result = [df.drop(columns=drop_columns, errors='ignore') for df in result]

    return result if len(result) > 1 else result[0]


def custom_melt(df: pd.DataFrame, variable_name, value_name):
    melted_df = pd.melt(df, var_name=variable_name, value_name=value_name)
    return melted_df


def convert_to_dataframe(data):
    try:
        # Convert to DataFrame if input is not already a DataFrame
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, (list, tuple, set)):
            return pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        elif isinstance(data, pd.Series):
            return pd.DataFrame({data.name: data})
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        else:
            raise ValueError("Unsupported data type. Please provide a supported data type.")
    except Exception as n:
        print(f"An error occurred during data conversion: {n}")
        return None


######### CLEANING DATA #########

def handle_nans_and_duplicates(dataframe: pd.DataFrame, target_column=None, output_folder=None):
    if target_column and dataframe[target_column].isna().any():
        raise ValueError(
            f"Target column '{target_column}' has NaN values. Please review the data.")

    nan_columns = dataframe.columns[dataframe.isna().any()]

    if nan_columns.any():
        cleaned_dataframe = dataframe.drop(columns=nan_columns)

        if output_folder:
            nan_output_path = os.path.join(output_folder, "nans.csv")
            nan_dataframe = dataframe[nan_columns]
            nan_dataframe.to_csv(nan_output_path, index=False)
            print(f"NaN values exported to: {nan_output_path}")
    else:
        cleaned_dataframe = dataframe

    duplicate_rows = cleaned_dataframe[cleaned_dataframe.duplicated()]

    if not duplicate_rows.empty:
        cleaned_dataframe = cleaned_dataframe.drop_duplicates()

        if output_folder:
            duplicates_output_path = os.path.join(output_folder, "duplicates.csv")
            duplicate_rows.to_csv(duplicates_output_path, index=False)
            print(f"Duplicate rows exported to: {duplicates_output_path}")

    return cleaned_dataframe


def function_remove_spaces(df: pd.DataFrame):
    df = df.applymap(lambda x: str(x).replace(' ', ''))
    return df


######### CLEANING DATA: specific functions #########


def remove_and_export_outliers(df: pd.DataFrame, outlier_threshold=None, target_column=None, output_folder=None):
    if outlier_threshold is None:
        return df

    columns_for_zscore = df.columns.difference([target_column]) if target_column else df.columns

    z_scores = np.abs(zscore(df[columns_for_zscore]))

    outliers = (z_scores > outlier_threshold).any(axis=1)

    if output_folder and outliers.any():
        outliers_output_path = os.path.join(output_folder, "outliers.csv")
        outliers_df = df[outliers]
        outliers_df.to_csv(outliers_output_path, index=False)
        print(f"Outliers exported to: {outliers_output_path}")

    df_no_outliers = df[~outliers]

    return df_no_outliers

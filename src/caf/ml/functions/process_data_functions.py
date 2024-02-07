# -*- coding: utf-8 -*-
"""
Created on: 16/11/2023
Original author: Adil Zaheer
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings


######### READ DATA FUNCTIONS #########

def read_folder(folder_path):
    dataframes_in = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            df = pd.read_csv(file_path, low_memory=False)
            dataframes_in[file_name] = df
    return dataframes_in


def read_csvs(x, y):
    x_in = pd.read_csv(x, low_memory=False)
    y_in = pd.read_csv(y, low_memory=False)
    return x_in, y_in


######### NUMERIC CONVERSION #########


def find_numeric_target_column(data, target_column):
    # todo fix this to work without target column
    if target_column is None:
        raise ValueError("Target column is None")

    # Check if the target column is present in the data
    if target_column in data.columns:
        # Check if the target column is numeric
        if not pd.to_numeric(data[target_column], errors='coerce').notna().all():
            # Try to convert the target column to numeric
            data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
            if not data[target_column].notna().all():
                raise ValueError(
                    "The target column could not be converted to numeric.")
            print(
                f"The target column '{target_column}' has been converted to numeric.")
        return data

    raise ValueError(
        "Target column not found in the data. Modeling may require a numeric target column.")


def process_data_numeric(data, keep_columns=None, output_path=None):
    print(data)
    # Convert to DataFrame if input is not dataframe
    if not isinstance(data, pd.DataFrame):
        data = convert_to_dataframe(data)

    # Convert remaining columns to numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Identify non-numeric columns
    non_numeric_columns = data.columns[~data.applymap(np.isreal).all()]

    # Set keep_columns to an empty set if not provided
    keep_columns = keep_columns or set()

    # Drop non-numeric columns, but only if they are not in the keep_columns set
    columns_to_drop = non_numeric_columns.difference(keep_columns)
    if columns_to_drop:
        print(f"Dropping non-numeric columns: {', '.join(columns_to_drop)}")

    data = data.drop(columns=columns_to_drop)

    # Export non-numeric columns to a separate file if output_path is provided
    if output_path:
        output_file_path = os.path.join(output_path, "non_numeric.csv")
        non_numeric_df = data[non_numeric_columns]
        non_numeric_df.to_csv(output_file_path, index=False)
        print(f"Non-numeric columns exported to: {output_file_path}")

    return data


######### SET DATAFRAME STRUCTURE #########


def index_sorter(*dataframes, index_columns=None, drop_columns=None):
    if not dataframes:
        raise ValueError("At least one dataframe must be provided")

    def set_index(df):
        if index_columns:
            return df.set_index(index_columns)
        else:
            return df

    result = [set_index(df) for df in dataframes]

    if drop_columns:
        result = [df.drop(columns=drop_columns, errors='ignore') for df in result]

    return result if len(result) > 1 else result[0]


def custom_melt(df, variable_name, value_name):
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


def handle_nans_and_duplicates(dataframe):
    # Find NaN values
    nan_columns = dataframe.columns[dataframe.isna().any()]

    # Check if there are any NaN values
    if nan_columns.any():
        # Drop columns with NaN values
        cleaned_dataframe = dataframe.drop(columns=nan_columns)
    else:
        # The DataFrame remains unchanged if there are no NaN values
        cleaned_dataframe = dataframe

    # Check for duplicate rows
    duplicate_rows = cleaned_dataframe[cleaned_dataframe.duplicated()]

    # Check if there are any duplicate rows
    if not duplicate_rows.empty:
        # Remove duplicate rows from the DataFrame
        cleaned_dataframe = cleaned_dataframe.drop_duplicates()

    return cleaned_dataframe


def function_remove_spaces(df: pd.DataFrame):
    df = df.applymap(lambda x: str(x).replace(' ', ''))
    return df


######### CLEANING DATA: specific functions #########


def remove_and_export_outliers(df, outlier_threshold=None):
    # Return the original DataFrame if no threshold is specified
    if outlier_threshold is None:
        return df

    # Calculate z-scores for each column
    z_scores = np.abs(zscore(df))

    # Identify outliers based on the threshold
    outliers = (z_scores > outlier_threshold).any(axis=1)

    # Separate outliers and non-outliers
    df_no_outliers = df[~outliers]
    return df_no_outliers


def assess_correlation(df, target_column, threshold=0.7, vif_threshold=10.0):
    # Exclude the target column from correlation analysis
    df_features = df.drop(columns=[target_column])

    # Calculate the correlation matrix for numeric columns
    correlation_matrix = df_features.corr()

    # Identify highly correlated columns
    highly_correlated_columns = correlation_matrix.columns[
        (correlation_matrix.abs() > threshold).any(axis=0)
    ].tolist()

    # Drop highly correlated columns
    df_no_correlation = df.drop(columns=highly_correlated_columns)

    # Check if there are variables left after dropping highly correlated ones
    if df_no_correlation.shape[1] == 0:
        # Issue a warning if no variables are left
        warnings.warn("No variables left after dropping highly correlated ones.", UserWarning)
        return df

    # Check for multicollinearity using VIF
    variables = df_no_correlation.columns
    vif_data = pd.DataFrame()
    vif_data["Variable"] = variables
    vif_data["VIF"] = [variance_inflation_factor(df_no_correlation.values.astype(float), i) for i in range(df_no_correlation.shape[1])]

    # Identify variables with high VIF
    high_vif_variables = vif_data[vif_data["VIF"] > vif_threshold]["Variable"].tolist()

    if high_vif_variables is not None:
        print(f"Columns removed due to high VIF: {high_vif_variables}")
        return df_no_correlation

    # Add the target column back to the modified dataframe
    df_no_correlation[target_column] = df[target_column]

    # If there are still variables left after correlation matrix evaluation, proceed with the modified dataframe
    return df_no_correlation

# -*- coding: utf-8 -*-
"""
Created on: 16/11/2023
Updated on:

Original author: Adil Zaheer
Last update made by:
Other updates made by:

File purpose: Process data to be ready for future modelling

"""
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor


class DataProcessor:
    def __init__(self, x, y, folder_path, index_columns, drop_columns, wide_format, variable_name, value_name,
                 keep_columns, target_column, outlier_threshold=None):
        final_data = None

        if x:
            x_ = pd.read_csv(x, low_memory=False)
            data = index_sorter(x_, index_columns=index_columns, drop_columns=drop_columns)
            final_data = function_remove_spaces(data)

            if wide_format is not None:
                x1 = custom_melt(x_, variable_name, value_name)
                data = index_sorter(x1, index_columns=[variable_name])
                final_data = function_remove_spaces(data)

            if x and y:
                x_, y_ = read_csvs(x, y)
                x1 = index_sorter(x_, index_columns=index_columns, drop_columns=drop_columns)
                y1 = index_sorter(y_, index_columns=index_columns, drop_columns=drop_columns)
                data = pd.merge(x1, y1, left_index=True, right_index=True, how="left")
                final_data = function_remove_spaces(data)

            if wide_format is not None:
                if x:
                    x_ = pd.read_csv(x, low_memory=False)
                    x1 = custom_melt(x_, variable_name, value_name)
                    data = index_sorter(x1, index_columns=[variable_name], drop_columns=drop_columns)
                    final_data = function_remove_spaces(data)

                if y:
                    y_ = pd.read_csv(y, low_memory=False)
                    y1 = custom_melt(y_, variable_name, value_name)
                    data = index_sorter(y1, index_columns=[variable_name], drop_columns=drop_columns)
                    final_data = function_remove_spaces(data)

                if x and y:
                    x_, y_ = read_csvs(x, y)
                    x1 = custom_melt(x_, variable_name, value_name)
                    y1 = custom_melt(y_, variable_name, value_name)
                    xfinal = index_sorter(x1, index_columns=[variable_name], drop_columns=drop_columns)
                    yfinal = index_sorter(y1, index_columns=[variable_name], drop_columns=drop_columns)
                    data = pd.merge(xfinal, yfinal, left_index=True, right_index=True, how="left")
                    final_data = function_remove_spaces(data)

        elif folder_path:
            dat = read_folder(folder_path)
            if wide_format is not None:
                dat_ = custom_melt(dat, variable_name, value_name)
                dat1 = {
                    key: index_sorter(df, index_columns=index_columns, drop_columns=drop_columns)
                    for key, df in dat_.items()
                }
                data = combine_data(dat1)
                final_data = function_remove_spaces(data)
            else:
                dat1 = {
                    key: index_sorter(df, index_columns=index_columns, drop_columns=drop_columns)
                    for key, df in dat.items()
                }
                data = combine_data(dat1)
                final_data = function_remove_spaces(data)

        self.data = final_data
        self.keep_columns = keep_columns
        self.outlier_threshold = outlier_threshold
        self.target_column = target_column

        self.data = self.find_numeric_target_column()
        self.data = self.process_data_numeric()
        self.data = self.handle_nans_and_duplicates()
        self.data = self.remove_and_export_outliers()
        self.data = self.assess_correlation()
        self.print_final_data_info()

    def find_numeric_target_column(self):
        return find_numeric_target_column(self.data, target_column=self.target_column)

    def process_data_numeric(self):
        return process_data_numeric(self.data, keep_columns=self.keep_columns)

    def handle_nans_and_duplicates(self):
        return handle_nans_and_duplicates(self.data)

    def remove_and_export_outliers(self):
        return remove_and_export_outliers(self.data, outlier_threshold=self.outlier_threshold)

    def assess_correlation(self):
        return assess_correlation(self.data, target_column=self.target_column)

    def print_final_data_info(self):
        print("Final Data Shape:")
        print(self.data.shape)

        print("Final Data Contents:")
        print(self.data)



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
                raise ValueError("The target column could not be converted to numeric.")
            print(f"The target column '{target_column}' has been converted to numeric.")
        return data

    raise ValueError("Target column not found in the data. Modeling may require a numeric target column.")


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


def combine_data(long_dataframes):
    complete_data = pd.concat(long_dataframes, axis=0)
    return complete_data


def custom_melt(df, variable_name, value_name):
    melted_df = pd.melt(df, var_name=variable_name, value_name=value_name)
    return melted_df


def function_remove_spaces(df: pd.DataFrame):
    df = df.applymap(lambda x: str(x).replace(' ', ''))
    return df


##########################


def process_data_numeric(data, keep_columns=None):
    print(data)
    # Convert to DataFrame if input is a Python array
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Convert remaining columns to numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Identify non-numeric columns
    non_numeric_columns = data.columns[~data.applymap(np.isreal).all()]

    # Set keep_columns to an empty set if not provided
    keep_columns = keep_columns or set()

    # Drop non-numeric columns, but only if they are not in the keep_columns set
    data = data.drop(columns=non_numeric_columns.difference(keep_columns))

    return data


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

    # Find and sort highly correlated variables
    correlated_columns = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_columns.add(colname)

    # Drop highly correlated cols
    df_no_correlation = df.drop(columns=correlated_columns, axis=1)

    # Check if there are variables left after dropping highly correlated ones
    if df_no_correlation.shape[1] == 0:
        print("No variables left after dropping highly correlated ones.")
        return df

    # Check for multicollinearity using VIF
    variables = df_no_correlation.columns
    vif_data = pd.DataFrame()
    vif_data["Variable"] = variables
    vif_data["VIF"] = [variance_inflation_factor(df_no_correlation.values.astype(float), i) for i in range(df_no_correlation.shape[1])]

    # Identify variables with high VIF
    high_vif_variables = vif_data[vif_data["VIF"] > vif_threshold]["Variable"].tolist()

    if high_vif_variables:
        print(f"Columns removed due to high VIF: {high_vif_variables}")
        return df_no_correlation

    # If there are no variables left after correlation matrix evaluation, continue with the original dataframe
    if df_no_correlation.shape[1] == 0:
        print("No variables left after dropping highly correlated ones.")
        return df

    # Add the target column back to the modified dataframe
    df_no_correlation[target_column] = df[target_column]

    # If there are still variables left after correlation matrix evaluation, proceed with the modified dataframe
    return df_no_correlation

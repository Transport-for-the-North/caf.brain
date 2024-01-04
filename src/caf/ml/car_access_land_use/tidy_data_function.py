# -*- coding: utf-8 -*-
"""
Created on: 12/22/2023
Updated on:

Original author: Adil Zaheer
Last update made by:
Other updates made by:

File purpose: Tidy data ready for machine learning modelling

"""
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor


def handle_nans_and_duplicates(dataframe, output_folder=None):
    # Find NaN values
    nan_matrix = dataframe.isna()

    # Check if there are any NaN values
    if nan_matrix.any().any():
        # Print message about NaN values
        print("NaN values found in the DataFrame.")

        # Save NaN values to a CSV file
        if output_folder:
            nan_output_path = Path(output_folder) / "nan_and_duplicate_values.csv"
            nan_matrix.to_csv(nan_output_path)
            print(f"NaN and duplicate values saved to: {nan_output_path}")

        # Remove NaN values from the DataFrame
        cleaned_dataframe = dataframe.dropna()

        # Print message about NaN removal
        print("NaN values removed from the DataFrame.")
    else:
        # Print message about no NaN values
        print("No NaN values found in the DataFrame. The DataFrame remains unchanged.")
        cleaned_dataframe = dataframe

    # Check for duplicate rows
    duplicate_rows = cleaned_dataframe[cleaned_dataframe.duplicated()]

    # Check if there are any duplicate rows
    if not duplicate_rows.empty:
        # Print message about duplicate values
        print("Duplicate rows found in the DataFrame.")

        # Save duplicate rows to a CSV file
        if output_folder:
            dup_output_path = Path(output_folder) / "nan_and_duplicate_values.csv"
            duplicate_rows.to_csv(dup_output_path, mode='a', header=False)
            print(f"Duplicate rows saved to: {dup_output_path}")

        # Remove duplicate rows from the DataFrame
        cleaned_dataframe = cleaned_dataframe.drop_duplicates()

        # Print message about duplicate removal
        print("Duplicate rows removed from the DataFrame.")
    else:
        # Print message about no duplicate rows
        print("No duplicate rows found in the DataFrame. The DataFrame remains unchanged.")

    return cleaned_dataframe


def remove_and_export_outliers(df, threshold=None, output_folder=None):
    # Return the original DataFrame if no threshold is specified
    if threshold is None:
        return df, None

    # Convert DataFrame to numeric (handle non-numeric values)
    numeric_df = df.apply(pd.to_numeric, errors='coerce')

    # Calculate Z-scores for each column
    z_scores = np.abs(zscore(numeric_df))

    # Identify outliers based on the threshold
    outliers = (z_scores > threshold).any(axis=1)

    # Separate outliers and non-outliers
    df_no_outliers = df[~outliers]
    df_outliers = df[outliers]

    # Export outliers to a CSV file if an output folder is specified
    if output_folder is not None and not df_outliers.empty:
        outliers_filepath = Path(output_folder) / "outliers.csv"
        df_outliers.to_csv(outliers_filepath, index=False)

    return df_no_outliers, df_outliers if not df_outliers.empty else None


def assess_correlation(df, threshold=0.7, vif_threshold=5.0):
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.columns.difference(numeric_columns)

    # Handle non-numeric columns separately
    non_numeric_df = df[non_numeric_columns]

    # Ensure all numeric columns are numeric
    numeric_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Replace infinite values with NaN and drop columns with NaN
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    # Calculate the correlation matrix for numeric columns
    correlation_matrix = numeric_df.corr()

    # Find and handle highly correlated variables
    correlated_columns = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_columns.add(colname)

    # Drop highly correlated numeric columns
    df_no_correlation = numeric_df.drop(columns=correlated_columns, axis=1)

    # Check for multicollinearity using VIF
    variables = df_no_correlation.columns
    vif_data = pd.DataFrame()
    vif_data["Variable"] = variables
    vif_data["VIF"] = [variance_inflation_factor(df_no_correlation.values.astype(float), i) for i in range(df_no_correlation.shape[1])]

    # Identify variables with high VIF
    high_vif_variables = vif_data[vif_data["VIF"] > vif_threshold]["Variable"].tolist()

    # Print a message about non-numeric columns
    if not non_numeric_columns.empty:
        print(f"Non-numeric columns ignored during correlation analysis: {non_numeric_columns.tolist()}")

    # Print a message about columns removed due to high VIF
    if high_vif_variables:
        print(f"Columns removed due to high VIF: {high_vif_variables}")

    # Concatenate numeric and non-numeric columns back together
    df_no_multicollinearity = pd.concat([df[non_numeric_columns], df_no_correlation], axis=1)

    return df_no_multicollinearity


def main_tdf(data, output_folder):

    cleaned_data = handle_nans_and_duplicates(data, output_folder=output_folder)
    result_no_outliers, result_outliers = remove_and_export_outliers(cleaned_data, threshold=2.5,
                                                                     output_folder=output_folder)

    improved_data = assess_correlation(result_no_outliers)
    print(improved_data)

    output_path = Path(output_folder) / "data.csv"
    improved_data.to_csv(output_path)

    return improved_data

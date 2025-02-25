# -*- coding: utf-8 -*-
"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
from pathlib import Path
from typing import Union, List, Any, Optional, Dict
import numpy as np
import pandas as pd
from caf.brain.ml.inputs_and_baseclasses.baseclasses import ValidateData
import logging
LOG = logging.getLogger(__name__)

class InitialDataProcessing:
    def __init__(self,
                 file_path: Path,
                 folder_path: Path,
                 output_path: Path,
                 target_column: str,
                 custom_index: List[str],
                 column_name_to_drop_rows: List[str],
                 value_in_row: List[str],
                 weight_column: str,
                 categorical_features: List[str],
                 numerical_features: List[str],
                 classification_prediction: tuple[int, ...],
                 is_test_data: bool = False):
        """
        Class for processing initial input data to the caf.brAIn prediction
        model.

        :param file_path: Optional path to data to be used for modelling.
        :param folder_path: Optional path to folder of data to be used for
                            modelling.
        :param output_path: Path to output location.
        :param target_column: sting column name of value to predict.
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
        :param is_test_data: Set to False by default unless test data is
                             being processed. This is set to true when using
                             the main prediction function. If using separately
                             then set to True or False when calling the class.
        """

        self.file_path = file_path
        self.folder_path = folder_path
        self.output_path = output_path
        self.target_column = target_column
        self.custom_index = custom_index
        self.column_name_to_drop_rows = column_name_to_drop_rows
        self.value_in_row = value_in_row
        self.weight_column = weight_column
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.classification_prediction = classification_prediction
        self.is_test_data = is_test_data

        self.df: pd.DataFrame = None
        self.dataframes: Dict[pd.DataFrame] = {}

    # # # Data flow pipelines # # #
    def execute_pipeline(self,
                         is_test_data: bool) -> Dict[str, pd.DataFrame]:
        """
        Runs the InitialDataProcessing pipeline.

        :return: dictionary containing the finished processed dataframes
        """
        self.read_data()
        self.process_dataframes(is_test_data)
        self.validate_data(is_test_data)
        return self.dataframes

    def data_already_split_pipeline(self,
                                    is_test_data: bool) -> Dict[str, pd.DataFrame]:
        """
        Runs the pipeline if train and test data is already split by the user
        or previous model runs.

        :return: dictionary containing the finished processed dataframes
        """
        self.process_dataframes(is_test_data)
        self.validate_data(is_test_data)
        return self.dataframes

    def read_data(self) -> None:
        """
        Read data from file or folder based on provided paths
        """
        if self.file_path:
            self.df = self.read_file(self.file_path)
            LOG.info(self.df)
        elif self.folder_path:
            self.dataframes = self.read_folder(self.folder_path)
            LOG.info(self.dataframes)
        else:
            LOG.error("Either file_path or folder_path must be provided")
            raise ValueError("Either file_path or folder_path must be provided")

    def process_dataframes(self, is_test_data: bool) -> None:
        """
        Process dataframe(s).
        """
        LOG.info(f"Processing {'test' if is_test_data is True else 'training'} data")

        if len(self.dataframes) > 1:
            if self.custom_index is None:
                raise ValueError("Must provide custom index in order to process \
                                  multiple dataframes. Data processing outside of \
                                  caf.ml is advised.")
            else:
                self.df = pd.concat(self.dataframes, axis=0)

        if self.df.empty:
            LOG.error(f"Dataframe {self.df} is empty")
            raise ValueError(f"Dataframe {self.df} is empty")

        df = self.convert_to_dataframe(self.df)

        target_column_ = [] if is_test_data else (
            [self.target_column] if isinstance(self.target_column, str) else [])
        weight_column_ = [self.weight_column] if isinstance(self.weight_column, str) else []
        custom_index = self.custom_index or []
        categorical_features = self.categorical_features or []
        numerical_features = self.numerical_features or []

        if self.numerical_features is None:
            columns_to_keep = custom_index + categorical_features + target_column_ + weight_column_
        elif self.categorical_features is None:
            columns_to_keep = custom_index + numerical_features + target_column_ + weight_column_
        else:
            columns_to_keep = custom_index + categorical_features + numerical_features + target_column_ + weight_column_

        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        df = df[columns_to_keep]

        if is_test_data:
            LOG.info("Processing test data - skipping target column operations")
            df = self.function_remove_spaces(df)

            if self.column_name_to_drop_rows and self.value_in_row:
                df = self.drop_rows(df,
                                    self.column_name_to_drop_rows,
                                    self.value_in_row)

            df = self.handle_nans_and_duplicates(df,
                                                 target_column=None,
                                                 output_folder=self.output_path)
            df = df.apply(pd.to_numeric, errors='coerce')

            if self.custom_index:
                df = self.index_sorter(df, self.custom_index)

        else:
            df = self.function_remove_spaces(df)

            if self.column_name_to_drop_rows and self.value_in_row:
                df = self.drop_rows(df,
                                    self.column_name_to_drop_rows,
                                    self.value_in_row)

            if self.target_column not in df.columns:
                LOG.error(f"Target column '{self.target_column}' not found in training data")
                raise ValueError(
                    f"Target column '{self.target_column}' not found in training data")

            df = self.handle_nans_and_duplicates(df,
                                                 target_column=self.target_column,
                                                 output_folder=self.output_path)

            df = self.numeric_transformation(df, self.target_column)

            if self.custom_index:
                df = self.index_sorter(df, self.custom_index)

            if self.classification_prediction:
                df = self.transform_target_column(df,
                                                  target_column=self.target_column,
                                                  classification_prediction=self.classification_prediction)

        df = self.convert_to_dataframe(df, columns=df.columns, index=df.index)

        filename = os.path.basename(self.file_path) if self.file_path else "processed_data"
        self.dataframes[filename] = df

    def validate_data(self, is_test_data) -> None:
        """
        Validate processed dataframes using the ValidateData class. Ensures
        data is in the format of the base class.
        """
        LOG.info("Starting data validation")
        for name, df in self.dataframes.items():
            validator = ValidateData(
                dataframe=df,
                custom_index=self.custom_index,
                target_column=self.target_column
            )

            if is_test_data is False:
                validator.index_present()
                validator.target_column_present()
                validator.explanatory_data()
                validator.is_data_numeric()
                validator.data_correct_shape()
            else:
                LOG.info("Test data detected - skipping target column validation")
                validator.index_present()
                validator.explanatory_data()
                validator.is_data_numeric()


    # # # DATA PROCESSING METHODS # # #
    @staticmethod
    def read_file(file_path) -> pd.DataFrame:
        """
        Reads in a file whilst trying all common encoding types and file types.
        :param file_path: file path location.
        :return: pandas dataframe.
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == '.csv':
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                        return df
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                raise UnicodeDecodeError(f"Unable to read CSV file with encodings: {encodings}")

            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                return df

            elif file_extension == '.json':
                df = pd.read_json(file_path)
                return df

            else:
                LOG.error(f"Unsupported file type: {file_extension}. Supported types: CSV, XLSX and JSON")
                raise ValueError(
                    f"Unsupported file type: {file_extension}. Supported types: CSV, XLSX and JSON")

        except Exception as e:
            LOG.error(f"Error reading file {file_path}: {e}")
            raise


    @staticmethod
    def read_folder(folder_path: Path) -> dict:
        """
        Creates a dictionary of dataframes based on a path. #TODO MAKE IT READ IN OTHER FILE TYPES

        :param folder_path: path to a folder that contains csvs to be used as
                            input data.
        :return: dictionary of dataframes.
        """
        dataframes_dict = {}
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".csv"):
                df = pd.read_csv(file_path, low_memory=False)
                dataframes_dict[file_name] = df
        return dataframes_dict

    @staticmethod
    def function_remove_spaces(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes any whitespace from column titles. A common problem in data.

        :param df: input dataframe.
        :return: dataframe with tidy column titles.
        """
        df = df.map(lambda x: str(x).replace(' ', ''))
        return df

    @staticmethod
    def index_sorter(df: pd.DataFrame,
                     custom_index: list[str]) -> pd.DataFrame:
        """
        Sets columns in a dataframe as the index.

        :param df: input dataframe.
        :param custom_index: list of strings that are columns in the dataframe.
                              These will be set as an index.
        :return: input dataframe with a custom index.
        """
        for col in custom_index:
            if col not in df.columns:
                LOG.error(f"Column '{col}' not found in DataFrame.")
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        df.set_index(custom_index, inplace=True, verify_integrity=False)

        return df

    @staticmethod
    def drop_rows(df: pd.DataFrame,
                  column_name_to_drop_rows: str,
                  value_in_row: Union[str, float, int]) -> pd.DataFrame:
        """
        Removes specified rows from the input dataframe.

        :param df: input data.
        :param column_name_to_drop_rows: Column names where the problematic
                                         rows exist.
        :param value_in_row: The value inside the row that should be removed.
        :return: dataframe with rows dropped.
        """
        df = df.astype(str)
        for col, val in zip(column_name_to_drop_rows, value_in_row):
            if col in df.columns:
                df = df[df[col] != val]
                LOG.info(f"Rows where {col} is {val} have been dropped.")
            else:
                LOG.warning(f"Column {col} does not exist in the DataFrame.")
        return df

    @staticmethod
    def handle_nans_and_duplicates(dataframe: pd.DataFrame,
                                   target_column: str = None,
                                   output_folder: Path = None) -> pd.DataFrame:
        """
        Removes and exports nans and duplicates.

        :param dataframe: input dataframe.
        :param target_column: column in the dataframe specified by user.
        :param output_folder: path to output folder.
        :return: dataframe without nans and duplicates.
        """
        if target_column and dataframe[target_column].isna().any():
            LOG.error(f"Target column '{target_column}' has NaN values. \
                               Please review the data.")
            raise ValueError(f"Target column '{target_column}' has NaN values. \
                               Please review the data.")

        rows_with_nans = dataframe[dataframe.isna().any(axis=1)]
        cleaned_dataframe = dataframe.dropna()

        if not rows_with_nans.empty and output_folder:
            nan_output_path = os.path.join(output_folder, "nans.csv")
            rows_with_nans.to_csv(nan_output_path, index=True)
            LOG.info(f"NaN rows exported to: {nan_output_path}")

        exact_duplicates = cleaned_dataframe[cleaned_dataframe.duplicated(keep=False)]

        if not exact_duplicates.empty:
            LOG.info(f"Exact duplicate rows found: {exact_duplicates}")

            if output_folder:
                duplicates_output_path = os.path.join(output_folder, "exact_duplicates.csv")
                exact_duplicates.to_csv(duplicates_output_path, index=True)
                LOG.info(f"Exact duplicate rows exported to: {duplicates_output_path}")

            cleaned_dataframe = cleaned_dataframe.drop_duplicates()

        return cleaned_dataframe

    @staticmethod
    def numeric_transformation(data: pd.DataFrame,
                               target_column: str) -> pd.DataFrame:
        """
        Ensures all data is numeric.

        :param data: input data.
        :param target_column: column in the dataframe specified by user.
        :return: Pandas dataframe.
        """
        if target_column not in data.columns:
            LOG.error("The target column is not in the dataframe. Please evaluate data.")
            raise ValueError(
                "The target column is not in the dataframe. Please evaluate data.")

        data = data.apply(pd.to_numeric, errors='coerce')

        if not pd.to_numeric(data[target_column], errors='coerce').notna().all():
            data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
            if not data[target_column].notna().all():
                LOG.error("The target column could not be converted to numeric.")
                raise ValueError(
                    "The target column could not be converted to numeric.")

        return data

    @staticmethod
    def convert_to_dataframe(data: Union[pd.DataFrame, List[Any], tuple, set,
                                   np.ndarray, pd.Series, dict],
                             columns: Optional[List[str]] = None,
                             index: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Converts data to a dataframe.

        :param data: input data of any regularly used formats.
        :param columns: list of column names to assign to dataframe.
        :param index: list of index labels to assign to index.
        :return: dataframe with correct index and column titles.
        """
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, (list, tuple, set)):
                df = pd.DataFrame(data)
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.Series):
                df = pd.DataFrame({data.name: data})
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                LOG.error("Unsupported data type. Please provide a supported data type.")
                raise ValueError("Unsupported data type. Please provide a supported data type.")

            if columns is not None:
                df.columns = columns
            if index is not None:
                df.index = index

            return df
        except Exception as n:
            LOG.error(f"An error occurred during data conversion: {n}")
            return None

    @staticmethod
    def transform_target_column(df, target_column, classification_prediction):
        if isinstance(classification_prediction, tuple) and len(classification_prediction) == 2:
            LOG.info(f'Binary model selected for values {classification_prediction}')
            df = df[df[target_column].isin(classification_prediction)]

        elif isinstance(classification_prediction, tuple) and len(classification_prediction) == 3:
            LOG.info(f'Multiclass model selected for values {classification_prediction}')
            df = df[df[target_column].isin(classification_prediction)]

        df[target_column] = df[target_column].astype(int)
        unique_values = df[target_column].unique()
        LOG.info(f"Unique values in {target_column} after transformation: {unique_values}")

        return df

# TODO: Custom outliers
# def remove_and_export_outliers(df: pd.DataFrame, outlier_threshold=None, target_column=None, output_folder=None):
#     if outlier_threshold is None:
#         return df
#
#     columns_for_zscore = df.columns.difference([target_column]) if target_column else df.columns
#
#     z_scores = np.abs(zscore(df[columns_for_zscore]))
#
#     outliers = (z_scores > outlier_threshold).any(axis=1)
#
#     if output_folder and outliers.any():
#         outliers_output_path = os.path.join(output_folder, "outliers.csv")
#         outliers_df = df[outliers]
#         outliers_df.to_csv(outliers_output_path, index=False)
#         print(f"Outliers exported to: {outliers_output_path}")
#
#     df_no_outliers = df[~outliers]
#
#     return df_no_outliers#



        # processed_dataframes = {}
        # for name, df in self.dataframes.items():
        #     if df.empty:
        #         print(f"Warning: Dataframe '{name}' is empty. Skipping processing.")
        #         processed_dataframes[name] = df
        #         continue
        #     df = self.convert_to_dataframe(df)
        #     print(
        #         f"Columns before processing '{name}': {df.columns}")
        #     target_column_ = [self.target_column] if isinstance(self.target_column, str) else []
        #     weight_column_ = [self.weight_column] if isinstance(self.weight_column, str) else []
        #     custom_index = self.custom_index or []
        #     categorical_features = self.categorical_features or []
        #     numerical_features = self.numerical_features or []
        #
        #     if self.numerical_features is None:
        #         columns_to_keep = custom_index + categorical_features + target_column_ + weight_column_
        #     elif self.categorical_features is None:
        #         columns_to_keep = custom_index + numerical_features + target_column_ + weight_column_
        #     else:
        #         columns_to_keep = custom_index + categorical_features + numerical_features + target_column_ + weight_column_
        #
        #     columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        #     df = df[columns_to_keep]
        #
        #     df = self.function_remove_spaces(df)
        #     if self.custom_index:
        #         df = self.index_sorter(df, self.custom_index)
        #
        #     if self.column_name_to_drop_rows and self.value_in_row:
        #         df = self.drop_rows(df,
        #                             self.column_name_to_drop_rows,
        #                             self.value_in_row)
        #
        #     df = self.handle_nans_and_duplicates(df,
        #                                          target_column=self.target_column,
        #                                          output_folder=self.output_path)
        #
        #     df = self.numeric_transformation(df, self.target_column)
        #     df = self.convert_to_dataframe(df, columns=df.columns, index=df.index)
        #     processed_dataframes[name] = df
        #
        # self.dataframes = processed_dataframes

# -*- coding: utf-8 -*-
"""
Created on: 16/11/2023
Original author: Adil Zaheer
"""
import pandas as pd
from caf.ml.functions.process_data_functions import (read_folder,
                                                     read_csvs,
                                                     find_numeric_target_column,
                                                     process_data_numeric,
                                                     index_sorter,
                                                     custom_melt,
                                                     handle_nans_and_duplicates,
                                                     function_remove_spaces,
                                                     remove_and_export_outliers,
                                                     assess_correlation,
                                                     convert_to_dataframe)


class DataProcessor:
    def __init__(self, x, y, folder_path, index_columns, drop_columns, wide_format, variable_name, value_name,
                 keep_columns, target_column, outlier_threshold=None):
        final_data = None

        if x:
            x_ = pd.read_csv(x, low_memory=False)
            data = index_sorter(x_, index_columns=index_columns, drop_columns=drop_columns)
            final_data = function_remove_spaces(data)
            final_data = convert_to_dataframe(final_data)

            if wide_format is not None:
                x1 = custom_melt(x_, variable_name, value_name)
                data = index_sorter(x1, index_columns=[variable_name])
                final_data = function_remove_spaces(data)
                final_data = convert_to_dataframe(final_data)

            if x and y:
                x_, y_ = read_csvs(x, y)
                x1 = index_sorter(x_, index_columns=index_columns, drop_columns=drop_columns)
                y1 = index_sorter(y_, index_columns=index_columns, drop_columns=drop_columns)
                data = pd.merge(x1, y1, left_index=True, right_index=True, how="left")
                final_data = function_remove_spaces(data)
                final_data = convert_to_dataframe(final_data)

            if wide_format is not None:
                if x:
                    x_ = pd.read_csv(x, low_memory=False)
                    x1 = custom_melt(x_, variable_name, value_name)
                    data = index_sorter(x1, index_columns=[variable_name], drop_columns=drop_columns)
                    final_data = function_remove_spaces(data)
                    final_data = convert_to_dataframe(final_data)

                if y:
                    y_ = pd.read_csv(y, low_memory=False)
                    y1 = custom_melt(y_, variable_name, value_name)
                    data = index_sorter(y1, index_columns=[variable_name], drop_columns=drop_columns)
                    final_data = function_remove_spaces(data)
                    final_data = convert_to_dataframe(final_data)

                if x and y:
                    x_, y_ = read_csvs(x, y)
                    x1 = custom_melt(x_, variable_name, value_name)
                    y1 = custom_melt(y_, variable_name, value_name)
                    xfinal = index_sorter(x1, index_columns=[variable_name], drop_columns=drop_columns)
                    yfinal = index_sorter(y1, index_columns=[variable_name], drop_columns=drop_columns)
                    data = pd.merge(xfinal, yfinal, left_index=True, right_index=True, how="left")
                    final_data = function_remove_spaces(data)
                    final_data = convert_to_dataframe(final_data)

        elif folder_path:
            dat = read_folder(folder_path)
            if wide_format is not None:
                dat_ = custom_melt(dat, variable_name, value_name)
                dat1 = {
                    key: index_sorter(df, index_columns=index_columns, drop_columns=drop_columns)
                    for key, df in dat_.items()
                }
                data = pd.concat(dat1, axis=0)
                final_data = function_remove_spaces(data)
                final_data = convert_to_dataframe(final_data)

            else:
                dat1 = {
                    key: index_sorter(df, index_columns=index_columns, drop_columns=drop_columns)
                    for key, df in dat.items()
                }
                data = pd.concat(dat1, axis=0)
                final_data = function_remove_spaces(data)
                final_data = convert_to_dataframe(final_data)

        self.data = final_data
        self.keep_columns = keep_columns
        self.outlier_threshold = outlier_threshold
        self.target_column = target_column

        self.data = self.find_numeric_target_column()
        self.data = self.process_data_numeric()
        self.data = self.handle_nans_and_duplicates()
        self.data = self.remove_and_export_outliers()
        self.data = self.assess_correlation()
        self.data = self.convert_to_dataframe()
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

    def convert_to_dataframe(self):
        return convert_to_dataframe(self.data)

    def print_final_data_info(self):
        print("Final Data Shape:")
        print(self.data.shape)

        print("Final Data Contents:")
        print(self.data)

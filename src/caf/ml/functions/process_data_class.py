# -*- coding: utf-8 -*-
"""
Created on: 16/11/2023
Original author: Adil Zaheer
"""
import os

import pandas as pd
from caf.ml.functions import (read_folder,
                              read_csvs,
                              find_numeric_target_column,
                              process_data_numeric,
                              index_sorter,
                              custom_melt,
                              function_remove_spaces,
                              remove_and_export_outliers,
                              convert_to_dataframe, drop_rows)


class DataProcessor:
    def __init__(self,
                 x,
                 y,
                 folder_path,
                 index_columns,
                 drop_columns,
                 keep_columns,
                 target_column,
                 output_folder,
                 wide_format,
                 variable_name,
                 value_name,
                 outlier_threshold,
                 categorical_target,
                 column_name_to_drop_rows,
                 value_in_row):
        print('----------------------------------------------')
        print('Data Processor is running')
        print('----------------------------------------------')
        final_data = None
        if x is not None and wide_format is None:
            x_ = pd.read_csv(x, low_memory=False)
            x_ = process_data_numeric(x_, keep_columns=keep_columns)
            data = index_sorter(x_, index_columns=index_columns, drop_columns=drop_columns)
            final_data = function_remove_spaces(data)
            final_data = convert_to_dataframe(final_data)

        elif x is not None and y is not None and wide_format is None:
            x_, y_ = read_csvs(x, y)
            x_ = process_data_numeric(x_, keep_columns=keep_columns)
            y_ = process_data_numeric(y_, keep_columns=keep_columns)
            x1 = index_sorter(x_, index_columns=index_columns, drop_columns=drop_columns)
            y1 = index_sorter(y_, index_columns=index_columns, drop_columns=drop_columns)
            data = pd.merge(x1, y1, left_index=True, right_index=True, how="left")
            final_data = function_remove_spaces(data)
            final_data = convert_to_dataframe(final_data)

        elif x is not None and wide_format is not None:
            x_ = pd.read_csv(x, low_memory=False)
            x_ = process_data_numeric(x_, keep_columns=keep_columns)
            x1 = custom_melt(x_, variable_name, value_name)
            data = index_sorter(x1, index_columns=[variable_name], drop_columns=drop_columns)
            final_data = function_remove_spaces(data)
            final_data = convert_to_dataframe(final_data)

        elif x is not None and y is not None and wide_format is not None:
            x_, y_ = read_csvs(x, y)
            x_ = process_data_numeric(x_, keep_columns=keep_columns)
            y_ = process_data_numeric(y_, keep_columns=keep_columns)
            x1 = custom_melt(x_, variable_name, value_name)
            y1 = custom_melt(y_, variable_name, value_name)
            xfinal = index_sorter(x1, index_columns=[variable_name], drop_columns=drop_columns)
            yfinal = index_sorter(y1, index_columns=[variable_name], drop_columns=drop_columns)
            data = pd.merge(xfinal, yfinal, left_index=True, right_index=True, how="left")
            final_data = function_remove_spaces(data)
            final_data = convert_to_dataframe(final_data)

        elif folder_path:
            dat = read_folder(folder_path)
            dat = process_data_numeric(dat, keep_columns=keep_columns)
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
        self.categorical_target = categorical_target
        self.column_name_to_drop_rows = column_name_to_drop_rows
        self.value_in_row = value_in_row
        self.index_columns = index_columns
        self.output_folder = output_folder

        self.data = self.find_numeric_target_column()
        #self.data = self.handle_nans_and_duplicates()
        self.data = self.remove_and_export_outliers()
        self.data = self.convert_to_dataframe()
        self.data = self.drop_rows()
        self.print_final_data_info()
        self.output_folder = output_folder
        self.output_processed_data()
        print('----------------------------------------------')
        print('Data Processor is finished')
        print('----------------------------------------------')


    def find_numeric_target_column(self):
        return find_numeric_target_column(self.data, target_column=self.target_column, categorical_target=self.categorical_target)

    #def handle_nans_and_duplicates(self):
    #    return handle_nans_and_duplicates(self.data, output_folder=self.output_folder)

    def remove_and_export_outliers(self):
        return remove_and_export_outliers(self.data, outlier_threshold=self.outlier_threshold)

    def drop_rows(self):
        return drop_rows(self.data,
                         column_name_to_drop_rows=self.column_name_to_drop_rows,
                         value_in_row=self.value_in_row)

    def convert_to_dataframe(self):
        return convert_to_dataframe(self.data)

    def print_final_data_info(self):
        print("Tidy_processed_data:")
        print(self.data.shape)
        print(self.data)

    def output_processed_data(self):
        output_filename = 'Tidy_processed_data.csv'
        output_path = os.path.join(self.output_folder, output_filename)
        self.data.to_csv(output_path, index=True)
        print('-------------------------------------------------------------')
        print(f"Tidy and processed data exported to: {output_path}")



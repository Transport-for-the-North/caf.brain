# -*- coding: utf-8 -*-
"""
Created on: 16/11/2023
Updated on:

Original author: Adil Zaheer
Last update made by:
Other updates made by:

File purpose: Process data to be ready for future modelling

"""
import pandas as pd
import os


class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def read_folder(folder_path):
        dataframes_in = {}
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".csv"):
                df = pd.read_csv(file_path, low_memory=False)
                dataframes_in[file_name] = df
        return dataframes_in

    @staticmethod
    def read_csvs(x, y):
        x_in = pd.read_csv(x, low_memory=False)
        y_in = pd.read_csv(y, low_memory=False)
        return x_in, y_in

    @staticmethod
    def index_sorter(*dataframes, index1=None, index2=None):
        if not dataframes:
            raise ValueError("At least one dataframe must be provided")

        def set_index(df):
            if index1 and index2:
                return df.set_index([index1, index2])
            elif index1:
                return df.set_index(index1)
            elif index2:
                return df.set_index(index2)
            else:
                return df

        result = [set_index(df) for df in dataframes]
        return result if len(result) > 1 else result[0]

    @staticmethod
    def combine_data(long_dataframes):
        complete_data = pd.concat(long_dataframes, axis=0)
        return complete_data

    @staticmethod
    def custom_melt(df, variable_name, value_name):
        melted_df = pd.melt(df, var_name=variable_name, value_name=value_name)
        return melted_df

    @staticmethod
    def function_remove_spaces(df: pd.DataFrame):
        df = df.applymap(lambda x: str(x).replace(' ', ''))
        return df

    def process_data(self, x, y, folder_path, index1, index2, wide_format, variable_name, value_name):
        final_data = None

        if x:
            x_ = pd.read_csv(x, low_memory=False)
            data = self.index_sorter(x_, index1=index1, index2=index2)
            final_data = self.function_remove_spaces(data)
            if wide_format is not None:
                x1 = self.custom_melt(x_, variable_name, value_name)
                data = self.index_sorter(x1, index1=variable_name)
                final_data = self.function_remove_spaces(data)
            if x and y:
                x_, y_ = self.read_csvs(x, y)
                x1 = self.index_sorter(x_, index1=index1, index2=index2)
                y1 = self.index_sorter(y_, index1=index1, index2=index2)
                data = pd.merge(x1, y1, left_index=True, right_index=True, how="left")
                final_data = self.function_remove_spaces(data)
            if wide_format is not None:
                if x:
                    x_ = pd.read_csv(x, low_memory=False)
                    x1 = self.custom_melt(x_, variable_name, value_name)
                    data = self.index_sorter(x1, index1=variable_name)
                    final_data = self.function_remove_spaces(data)
                if y:
                    y_ = pd.read_csv(y, low_memory=False)
                    y1 = self.custom_melt(y_, variable_name, value_name)
                    data = self.index_sorter(y1, index1=variable_name)
                    final_data = self.function_remove_spaces(data)
                if x and y:
                    x_, y_ = self.read_csvs(x, y)
                    x1 = self.custom_melt(x_, variable_name, value_name)
                    y1 = self.custom_melt(y_, variable_name, value_name)
                    xfinal = self.index_sorter(x1, index1=variable_name)
                    yfinal = self.index_sorter(y1, index1=variable_name)
                    data = pd.merge(
                        xfinal, yfinal, left_index=True, right_index=True, how="left"
                    )
                    final_data = self.function_remove_spaces(data)
        elif folder_path:
            dat = self.read_folder(folder_path)
            if wide_format is not None:
                dat_ = self.custom_melt(dat, variable_name, value_name)
                dat1 = {
                    key: self.index_sorter(df, index1=index1, index2=index2)
                    for key, df in dat_.items()
                }
                data = self.combine_data(dat1)
                final_data = self.function_remove_spaces(data)
            else:
                dat1 = {
                    key: self.index_sorter(df, index1=index1, index2=index2)
                    for key, df in dat.items()
                }
                data = self.combine_data(dat1)
                final_data = self.function_remove_spaces(data)
        print(final_data)
        return final_data



# init should be the thing that processes and sorts the data
# the methods are then different ways of exporting the sorted data based upon specific need
# class should call the functions in tidy_data,
# while class contains the data, do the things that need to be done to the data

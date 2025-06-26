# -*- coding: utf-8 -*- LOADING
"""
Created on: 04/09/2023
Updated on: 08/09/2023

Original author: Isaac Scott
Last update made by: Isaac Scott
Other updates made by: Adil Zaheer

File purpose: The file feeds into the land value uplift model. It contains
functions_to_be_processed that load, process and transform data for future modelling purposes.

"""
import os
from dataclasses import dataclass
from functools import reduce
from typing import Union
import pandas as pd
from pathlib import Path

import inputs


@dataclass
class XVar:
    """
    Summary
    -------
    Class for storing data about x_variables for the LVU model.

    This can be used either for data to be read in, in which case dataframe will be a path,
    or for data which has been read in in which case dataframe will be a pandas dataframe.

    Parameters
    ----------
    geog_col: name of the column containing geographic info, e.g. zone code.
    year_col: Name of the column containing the year of the data.
    val_name: Desired or existing name of column data is contained in
    dataframe: Either a path to a csv or a pandas dataframe containing data.
    """

    geog_col: str
    year_col: str
    val_name: str
    dataframe: Union[Path, pd.DataFrame]

    def load_custom_file(self):
        """
        The function loads csv's from a file path specified by the user. These
        csv's are processed and combined based upon their geographical
        properties. It essentially converts a path to a dataframe for
        self.dataframe.
        """

        df = pd.read_csv(self.dataframe)
        df.set_index(self.geog_col, inplace=True)
        df = interpolate(df)
        df = df.stack().reset_index()
        df.columns = [self.geog_col, self.year_col, self.val_name]
        return df.set_index([self.geog_col, self.year_col])


def read_multiple_files(folder_path):
    """Function is used to read in multiple files from a specific file path.
    This will be the location of the historical x data"""
    dataframes = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            df = pd.read_csv(file_path)
            dataframes[file_name] = XVar(
                geog_col="LA_code",
                year_col="year",
                val_name=file_name.split(".")[0],
                dataframe=df,
            )
    return dataframes


def process_ntem(
    params: inputs.NtemInputInfo,
):
    """
    This function processes the ntem data for future modelling. A list is returned
    which contains the processed data in the model appropriate format. The
    data path is specified outside the function by the user.
    """
    ind_cols = list(range(len(params.cat_cols) + 1))
    ntem_data = pd.read_csv(params.data_path, index_col=ind_cols)
    ntem_data.columns = ntem_data.columns.astype(int)
    ntem_data = interpolate(ntem_data).reset_index()
    out = []
    for i in params.cat_cols:
        df = ntem_data.groupby([params.zone_col, i]).sum()
        # Drop all columns which aren't the current category
        drop_cols = [j for j in params.cat_cols if j not in [i]]
        df = df.drop(drop_cols, axis=1).stack().unstack(level=i)
        if params.zone_col_rename is not None:
            df.index.names = [params.zone_col_rename, "year"]
        else:
            df.index.names = [params.zone_col, "year"]
        for col in df.columns:
            out_df = df[col]
            out.append(
                XVar(
                    geog_col=params.zone_col,
                    year_col="year",
                    val_name=col,
                    dataframe=out_df,
                )
            )
    return out


def interpolate(df: pd.DataFrame, non_year_cols: list[str] = None):
    """
    This function is used across other functions_to_be_processed within the loading.py script.
    It conducts linear interpolation to infill missing values in the years
    columns where applicable.
    """
    # Don't alter the input variable
    df = df.copy()
    if non_year_cols is not None:
        df = df.set_index(non_year_cols)
    df.columns = df.columns.astype(int)
    years = df.columns.sort_values()
    for ind, year in enumerate(years[:-1]):
        gap = years[ind + 1] - year
        if gap > 1:
            increment = (df[years[ind + 1]] - df[year]) / gap
            for i in range(1, gap):
                df[year + i] = df[year] + increment * i
    return df


def x_variable_processing(dataframes: list[XVar]):
    """This function processes the files in the dataframe dictionary produced
    by the read_multiple_files function. The dataframe produced (merged_df)
    contains both the x data and the y data in one dataframe in the best
    format for the model to forecast."""
    # Convert the dataframes from wide to long format and store them in a list
    melted_dataframes = []

    for var in dataframes:
        if isinstance(var.dataframe, pd.Series):
            var.dataframe = var.dataframe.to_frame()
            var.dataframe.columns = [var.val_name]
        df = var.dataframe.reset_index()
        melted_df = df.set_index([var.geog_col, var.year_col])
        melted_dataframes.append(melted_df)

    # Merge the dataframes together
    merged_df = reduce(
        lambda left, right: left.join(right),
        melted_dataframes,
    )
    merged_df = merged_df.replace(",", "")

    return merged_df


def tuning_features(merged_df):
    """This function contains any specific transformations that must be
    applied to either all data or specific data in merged_df. Feature scaling
    is initially done which is done to get a constant variance and average
    mean of all explanatory data."""
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(merged_df)

    # done in this way to ensure non-scaled variables are preserved
    scaled_df = pd.DataFrame(scaled_data, columns=merged_df.columns, index=merged_df.index)

    return scaled_df

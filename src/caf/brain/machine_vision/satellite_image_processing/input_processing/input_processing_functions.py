# -*- coding: utf-8 -*-
"""
Created on: 4/17/2025
Original author: Adil Zaheer
"""
# Built-Ins
import logging

# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
from pathlib import Path

# Third Party
import geopandas as gpd
import pandas as pd

# Local Imports
from caf.brain.machine_vision.satellite_image_processing.html_processing.html_processing_main import (
    main_process_html,
)
from caf.brain.machine_vision.satellite_image_processing.image_processing.image_processing_functions import (
    euclidean_distance,
)

LOG = logging.getLogger(__name__)


def process_coordinates(
    geo_df: gpd.geodataframe,
    df: pd.DataFrame,
    image_folder: Path,
    final_html_info: Path,
    x_coordinate: str,
    y_coordinate: str,
    output_path: Path,
    folder_if_loop: Path,
):
    """
    need to provide one of geo_df or df and one of image_folder or final_html_info
    first part generates satellite image information
    second part processes your coordinates to model
    euclidean_distance then calculated to find the most appropriate satelite image for your coordinate
    """
    # todo, add coordinate conversion functionality
    # todo, need to get a list of every satelite image

    file_name = "final_coordinate_data.csv"
    if os.path.exists(os.path.join(folder_if_loop, file_name)):
        final_coordinate_data = pd.read_csv(os.path.join(folder_if_loop, file_name))

        unique_box_boundaries = final_coordinate_data["box_boundary"].unique()
        unique_box_boundaries = pd.DataFrame(unique_box_boundaries, columns=["BNG_tile_names"])

        return final_coordinate_data, unique_box_boundaries

    if image_folder is None:
        raise ValueError(
            "Ensure that a path to an image folder is provided \
                          if image_info is None. This can be the outer folder \
                          that contains all the JPG images. This is used to \
                          create a csv of image information."
        )
    else:
        data_dict, final_html = main_process_html(
            folder_path=image_folder, output_path=output_path
        )
    # at this point, got html info for every tile we have:
    # box_boundary
    # latitude_wgs84
    # longitude_wgs84
    # tile_easting
    # tile_northing

    if geo_df is not None:
        if "geometry" not in geo_df.columns:
            raise ValueError("Ensure that geometry column is present in Geography data.")
        df["coordinates_easting"] = df["geometry"].x
        df["coordinates_northing"] = df["geometry"].y

    elif df is not None:
        df = df.rename(
            columns={x_coordinate: "coordinates_easting", y_coordinate: "coordinates_northing"}
        )

    else:
        raise ValueError(
            "Must provide one of geo_df or df. If geo_df then it \
                         should contain a geometry column. If df then the csv should \
                         contain two coordinates columns."
        )

    coordinate_data, unique_box_boundaries = euclidean_distance(df_a=final_html, df_b=df)

    final_coordinate_data = pd.merge(
        coordinate_data, final_html, on="box_boundary", how="inner"
    )

    final_coordinate_data.to_csv(os.path.join(folder_if_loop, file_name), index=False)

    return final_coordinate_data, unique_box_boundaries


def get_file_type(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".csv":
        return "csv"
    elif ext == ".shp":
        return "shp"
    else:
        return None


def process_file(file_path):
    file_type = get_file_type(file_path)
    if file_type == "csv":
        df = pd.read_csv(file_path)
        return df, file_type
    elif file_type == "shp":
        gdf = gpd.read_file(file_path)
        return gdf, file_type
    else:
        raise ValueError("Unsupported file type.")

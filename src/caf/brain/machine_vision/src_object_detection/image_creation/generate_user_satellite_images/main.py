"""
Created on: 9/25/2025
Original author: Adil Zaheer
"""

import pandas as pd
from pathlib import Path
import logging
from caf.brain.machine_vision.src_object_detection.image_creation.generate_user_satellite_images.functions import (
    _euclidean_distance,
    _user_image_path_finder, _TEMP_user_image_path_finder,
)
import geopandas as gpd

LOG = logging.getLogger(__name__)


def locate_user_coordinates(
    df: gpd.GeoDataFrame,
    output_path: Path,
    satellite_image_metadata: pd.DataFrame,
    image_folder_path: Path,
):
    """
    Locate the satellite images that correspond to user input coordinates.

    Parameters
    ----------
    df: gpd.GeoDataFrame that contains user coordinates.
    output_path: Path to output folder.
    satellite_image_metadata: Satellite image metadata containing tile names,
                              midpoints and path locations.
    image_folder_path: Path to folder that contains British National Grid tile
                       jpegs and their accompanying xml files.

    Returns
    -------
    user_image_metadata: Dataframe with user coordinate locations, relevant
                         classification and location specific satellite
                         metadata.
    """
    output_path = Path(output_path)
    image_folder_path = Path(image_folder_path)

    df["coordinates_easting"] = df["geometry"].x
    df["coordinates_northing"] = df["geometry"].y

    user_data_with_closest_tile = _euclidean_distance(df_a=satellite_image_metadata, df_b=df)

    path_df = _user_image_path_finder(
        image_folder=image_folder_path, user_coordinate_data=user_data_with_closest_tile
    )

    user_image_metadata = pd.merge(
        user_data_with_closest_tile, path_df, on="box_boundary", how="inner"
    )

    user_coordinate_filename = output_path / "user_coordinate_data.csv"
    user_image_metadata.to_csv(user_coordinate_filename, index=False)

    return user_image_metadata


def TEMP_locate_user_coordinates(
    df: gpd.GeoDataFrame,
    output_path: Path,
    satellite_image_metadata: pd.DataFrame,
    image_folder_path_north: Path,
    image_folder_path_south: Path
):
    """
    Locate the satellite images that correspond to user input coordinates.

    Parameters
    ----------
    df: gpd.GeoDataFrame that contains user coordinates.
    output_path: Path to output folder.
    satellite_image_metadata: Satellite image metadata containing tile names,
                              midpoints and path locations.
    image_folder_path: Path to folder that contains British National Grid tile
                       jpegs and their accompanying xml files.

    Returns
    -------
    user_image_metadata: Dataframe with user coordinate locations, relevant
                         classification and location specific satellite
                         metadata.
    """
    output_path = Path(output_path)
    image_folder_path_north = Path(image_folder_path_north)
    image_folder_path_south = Path(image_folder_path_south)

    df["coordinates_easting"] = df["geometry"].x
    df["coordinates_northing"] = df["geometry"].y

    user_data_with_closest_tile = _euclidean_distance(df_a=satellite_image_metadata, df_b=df)

    path_df = _TEMP_user_image_path_finder(
        image_folder_path_north=image_folder_path_north,
        image_folder_path_south=image_folder_path_south,
        user_coordinate_data=user_data_with_closest_tile
    )

    user_image_metadata = pd.merge(
        user_data_with_closest_tile, path_df, on="box_boundary", how="inner"
    )

    user_coordinate_filename = output_path / "user_coordinate_data.csv"
    user_image_metadata.to_csv(user_coordinate_filename, index=False)

    return user_image_metadata

"""
Created on: 9/25/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
from pathlib import Path

# Third Party
import geopandas as gpd
import pandas as pd

# Local Imports
from caf.brain.object_detection._functions.generate_user_image_metadata.functions import (
    _euclidean_distance,
    _user_image_path_finder,
)

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
    df:
        gpd.GeoDataFrame that contains user coordinates.
    output_path:
        Path to output folder.
    satellite_image_metadata:
        Satellite image metadata containing tile names, midpoints and path
        locations. Generated from image_info_generation.
    image_folder_path:
        Path to folder that contains British National Grid tile jpegs and
        their accompanying XML files.

    Returns
    -------
    user_image_metadata:
        Dataframe with user coordinate locations, relevant classification and
        location specific satellite metadata.
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

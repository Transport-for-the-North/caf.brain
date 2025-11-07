"""
Created on: 9/25/2025
Original author: Adil Zaheer
"""

import pandas as pd
from pathlib import Path
import logging
from caf.brain.machine_vision.src_object_detection.image_creation.generate_user_satellite_images.functions import (
    _read_path,
    _euclidean_distance,
    _user_image_path_finder,
)

LOG = logging.getLogger(__name__)


def locate_user_coordinates(
    user_locations_csv_path: Path,
    output_path: Path,
    satellite_image_metadata: pd.DataFrame,
    image_folder_path: Path,
    x_coordinate: str | None = None,
    y_coordinate: str | None = None,
):
    """
    Locate the satellite images that correspond to user input coordinates.

    Parameters
    ----------
    user_locations_csv_path: Path to user data that must contain coordinates
                             in either coordinate columns
                             (x_coordinate & y_coordinate) or a geography
                             column inside a shapefile.
    output_path: Path to output folder.
    satellite_image_metadata: Satellite image metadata containing tile names,
                              midpoints and path locations.
    image_folder_path: Path to folder that contains British National Grid tile
                       jpegs and their accompanying xml files.
    x_coordinate: X coordinate column inside your data if data is not a
                  shapefile.
    y_coordinate: Y coordinate column inside your data if data is not a
                  shapefile.

    Returns
    -------
    user_image_metadata: Dataframe with user coordinate locations, relevant
                         classification and location specific satellite
                         metadata.
    """
    output_path = Path(output_path)
    user_locations_csv_path = Path(user_locations_csv_path)
    image_folder_path = Path(image_folder_path)

    user_coordinate_filename = output_path / "user_coordinate_data.csv"

    if user_coordinate_filename.exists():
        df = pd.read_csv(user_coordinate_filename)
        return df

    if not user_locations_csv_path.exists():
        raise ValueError(
            "Please provide a csv path with your coordinates. \
                          These can be part of a geography column or two \
                          separate x and y columns"
        )

    df, file_type = _read_path(file_path=user_locations_csv_path)
    if file_type == "shp":
        if "geometry" not in df.columns:
            raise ValueError("Ensure that geometry column is present in shapefile data.")

        df["coordinates_easting"] = df["geometry"].x
        df["coordinates_northing"] = df["geometry"].y
    else:
        if x_coordinate and y_coordinate not in df.columns:
            raise ValueError(
                "Please provide the column name for your x coordinates \
                              and y coordinates in your data. Alternatively pass \
                              a shapefile with a geometry column"
            )

        df = df.rename(
            columns={x_coordinate: "coordinates_easting", y_coordinate: "coordinates_northing"}
        )

    user_data_with_closest_tile = _euclidean_distance(df_a=satellite_image_metadata, df_b=df)

    path_df = _user_image_path_finder(
        image_folder=image_folder_path, user_coordinate_data=user_data_with_closest_tile
    )

    user_image_metadata = pd.merge(
        user_data_with_closest_tile, path_df, on="box_boundary", how="inner"
    )

    user_image_metadata.to_csv(user_coordinate_filename, index=False)

    return user_image_metadata

# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 4/10/2025
Original author: Adil Zaheer
"""
# Built-Ins
import os
from pathlib import Path

# Third Party
import geopandas as gpd
import pandas as pd

# Local Imports
from caf.brain.machine_vision.satellite_image_processing.image_processing.image_processing_functions import (
    create_satellite_image_metadata,
    directory_iterator,
)
from caf.brain.machine_vision.satellite_image_processing.image_processing.main_image_crop import (
    image_crop_main,
)
from caf.brain.machine_vision.satellite_image_processing.input_processing.input_processing_functions import (
    process_coordinates,
)


def main_input_processing(
    geo_df: gpd.geodataframe,
    df: pd.DataFrame,
    image_folder: Path,
    final_html_info: Path,
    x_coordinate: str,
    y_coordinate: str,
    output_path: Path,
    folder_if_loop: Path,
):
    # todo make a flow for generating training images only
    # todo make one for running the entire model? this would be in a main elsewhere?
    # generating training images info just needs this info below?

    coordinate_df, unique_box_boundaries = process_coordinates(
        geo_df=geo_df,
        df=df,
        image_folder=image_folder,
        final_html_info=final_html_info,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
        output_path=output_path,
        folder_if_loop=folder_if_loop,
    )

    path_list_of_your_coordinates = directory_iterator(
        dir_path=image_folder,
        training_data=coordinate_df,
        image_reference_column="box_boundary",
        output=output_path,
        folder_if_loop=folder_if_loop,
    )

    metadata = create_satellite_image_metadata(dir_path=image_folder, output=output_path)

    image_crop_main(
        path_list=path_list_of_your_coordinates,
        image_supporting_data=coordinate_df,
        output=output_path,
        image_folder=image_folder,
        satellite_metadata=metadata,
        folder_if_loop=folder_if_loop,
    )

    return

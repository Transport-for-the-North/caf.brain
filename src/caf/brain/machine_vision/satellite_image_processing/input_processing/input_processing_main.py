# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 4/10/2025
Original author: Adil Zaheer
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path
from caf.brain.machine_vision.satellite_image_processing.image_processing.image_processing_functions import (directory_iterator,
                                                                                                             create_satellite_image_metadata)
from caf.brain.machine_vision.satellite_image_processing.image_processing.main_image_crop import image_crop_main
from caf.brain.machine_vision.satellite_image_processing.input_processing.input_processing_functions import process_coordinates


def main_input_processing(geo_df: gpd.geodataframe,
                          df: pd.DataFrame,
                          image_folder: Path,
                          final_html_info: Path,
                          x_coordinate: str,
                          y_coordinate: str,
                          output_path: Path):
    # todo make a flow for generating training images only
    # todo make one for running the entire model? this would be in a main elsewhere?
    # generating training images info just needs this info below?
    coordinate_df, unique_box_boundaries = process_coordinates(geo_df,
                                                               df,
                                                               image_folder,
                                                               final_html_info,
                                                               x_coordinate,
                                                               y_coordinate,
                                                               output_path)

    path_list_of_your_coordinates = directory_iterator(dir_path=image_folder,
                                                       training_data=coordinate_df,
                                                       image_reference_column='box_boundary',
                                                       output=output_path)

    metadata = create_satellite_image_metadata(dir_path=image_folder,
                                               output=output_path)

    image_crop_main(path_list=path_list_of_your_coordinates,
                    image_supporting_data=coordinate_df,
                    output=output_path,
                    image_folder=image_folder,
                    satellite_metadata=metadata)

    return

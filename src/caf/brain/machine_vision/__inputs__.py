# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 3/3/2025
Original author: Adil Zaheer
"""
from pathlib import Path
from typing import Optional
from caf.toolkit import BaseConfig


class GenerateSatelliteImagesInput(BaseConfig):
    # Model outputs
    output_path: Optional[Path] = None

    # Generate training images based on NoHam 2023 base network
    # Either training_data_information is required or both noham_path and geo_path
    generate_training_images: Optional[bool] = False
    training_data_information: Optional[Path] = None
    noham_path: Optional[Path] = None
    geo_path: Optional[Path] = None

    image_folder_path: Optional[Path] = None  # path to outermost satellite image folder
    final_html_info: Optional[Path] = None  # path to satellite image info (will be generated if it doesn't exist)

    # only relevant if noham_path and geo_path are none. The columns easting and northing coordinates for what you're trying to predict
    x_column: Optional[str] = None
    y_column: Optional[str] = None

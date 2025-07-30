"""
Created on: 3/3/2025
Original author: Adil Zaheer
"""

from pathlib import Path
from typing import Optional
from caf.toolkit import BaseConfig


class NoHAMInputs(BaseConfig):
    # only process noham data
    noham_db_path: Optional[Path] = None
    noham_shp_path: Optional[Path] = None
    output_path: Optional[Path] = None


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
    final_html_info: Optional[Path] = (
        None  # path to satellite image info (will be generated if it doesn't exist)
    )

    # only relevant if noham_path and geo_path are none. The columns easting and northing coordinates for what you're trying to predict
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    individual_junc_types: Optional[Path] = None


class ObjectDetectionInputs(BaseConfig):
    output: Optional[Path] = None
    class_names: Optional[list[str]] = None
    hyperparameter_optimisation: Optional[str] = None
    image_path: Optional[Path] = None
    path_to_code: Optional[Path] = (
        None  # path to where your brain folder is inside caf.brAIn e.g. C:\Users\Liberty\Documents\GitHub\caf.brain\src\caf\brain
    )


class ObjectDetectionSatellitesInputs(BaseConfig):
    """
    Main inputs for object detection using satellite images.
    """
    output_path: Optional[Path] = None
    image_folder_path: Optional[Path] = None
    satellite_image_metadata: Optional[Path] = None
    user_locations_path: Optional[Path] = None
    x_coordinate: Optional[str] = None
    y_coordinate: Optional[str] = None
    class_names: Optional[list[str]] = None
    hyperparameter_optimisation: Optional[str] = None
    prediction_images_folder: Optional[Path] = None

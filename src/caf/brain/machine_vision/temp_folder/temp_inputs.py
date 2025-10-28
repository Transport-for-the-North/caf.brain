"""
Created on: 10/10/2025
Original author: Adil Zaheer
"""

from pathlib import Path
from typing import Optional
from caf.toolkit import BaseConfig


class object_detection_inputs(BaseConfig):
    """"""

    output_path: Optional[Path] = None

    class ImageGenerationInputs(BaseConfig):
        """"""

        generate_images: Optional[bool] = False
        image_folder_path: Optional[Path] = None
        user_locations_csv_path: Optional[Path] = None
        x_coordinate: Optional[str] = None
        y_coordinate: Optional[str] = None

    class BuildModelInputs(BaseConfig):
        """"""

        build_model: Optional[bool] = False
        user_images_folder_path: Optional[Path] = None
        classification_names: Optional[list[str]] = None
        hyperparameter_optimisation: Optional[bool] = False

    image_generation_inputs: ImageGenerationInputs = ImageGenerationInputs()
    build_model_inputs: BuildModelInputs = BuildModelInputs()

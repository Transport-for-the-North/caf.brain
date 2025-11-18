"""
Created on: 10/10/2025
Original author: Adil Zaheer
"""

from pathlib import Path
from typing import Optional
from caf.toolkit import BaseConfig


class ObjectDetectionInputs(BaseConfig):
    """
    Input base class for caf.brAIn machine vision.

    """

    output_path: Optional[Path] = None
    user_locations_csv_path: Optional[Path] = None

    class ImageGenerationInputs(BaseConfig):
        """
        Inputs for training image generation.
        """

        generate_images: Optional[bool] = False
        image_folder_path: Optional[Path] = None
        x_coordinate: Optional[str] = None
        y_coordinate: Optional[str] = None

    class BuildModelInputs(BaseConfig):
        """
        Inputs for building a YOLO object detection model.
        """

        build_model: Optional[bool] = False
        user_images_folder_path: Optional[Path] = None
        classification_names: Optional[list[str]] = None
        hyperparameter_optimisation: Optional[str] = None

    class PredictionInputs(BaseConfig):
        """
        Inputs for object detection prediction.
        """

        prediction_object_detection: Optional[bool] = False
        trained_model: Optional[Path] = None

    image_generation_inputs: ImageGenerationInputs = ImageGenerationInputs()
    build_model_inputs: BuildModelInputs = BuildModelInputs()
    prediction_inputs: PredictionInputs = PredictionInputs()

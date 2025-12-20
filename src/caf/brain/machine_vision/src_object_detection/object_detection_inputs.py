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

    class ImageGenerationInputs(BaseConfig):
        """
        Inputs for training image generation.
        """
        generate_images: Optional[bool] = False
        image_folder_path: Optional[Path] = None
        img_locations_path: Optional[Path] = None

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
        os_path: Path | str | list[str] | None = None
        locations: Optional[str] = None
        model_path: Optional[Path] = None
        img_locations_pred: Path | str | list[str] | None = None

    class TempInputsWorkAround(BaseConfig):
        north_eng_images: Optional[Path] = None
        south_eng_images: Optional[Path] = None

    image_generation_inputs: ImageGenerationInputs = ImageGenerationInputs()
    build_model_inputs: BuildModelInputs = BuildModelInputs()
    prediction_inputs: PredictionInputs = PredictionInputs()
    temp_inputs: TempInputsWorkAround = TempInputsWorkAround()

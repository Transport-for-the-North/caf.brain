"""
Created on: 13/03/2026
Original author: Adil Zaheer
"""

# Built-Ins
from pathlib import Path
from typing import Optional

# Third Party
from caf.toolkit import BaseConfig


class PredictionInputs(BaseConfig):
    """
    Object detection inputs that are paths to external files.
    """

    user_location: Optional[Path | str] = None
    output_path: Optional[Path] = None
    image_folder_path: Optional[Path] = None
    model_path: Optional[Path] = None


class ObjectDetectionInputs(BaseConfig):
    """
    Caf.brAIn object detection model inputs.
    """

    object_detection: PredictionInputs = PredictionInputs()

"""
Created on: 06/11/2025
Original author: Adil Zaheer
"""

from pathlib import Path
import yaml
from caf.toolkit import ToolDetails, LogHelper
from caf.brain.machine_vision.src_object_detection.object_detection_main import main
from caf.brain.machine_vision.src_object_detection.object_detection_inputs import (
    ObjectDetectionInputs,
)


def model_setup():
    """
    Sets up logging, output folder and run config for the caf.brAIn object
    detection model.

    Returns
    -------
    None
    """
    with open("run_config.yaml", "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    params = ObjectDetectionInputs(**config_data)

    output_path = Path(params.output_path) / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    path = output_path / "log_file.log"
    details = ToolDetails("caf.brAIn object detection", "1.0.0")

    with LogHelper("caf.brain", details, console=True, log_file=path):
        main(params, output_path)


if __name__ == "__main__":
    model_setup()

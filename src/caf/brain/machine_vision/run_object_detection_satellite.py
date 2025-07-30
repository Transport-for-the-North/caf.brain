"""
Created on: 3/3/2025
Original author: Adil Zaheer
"""

import os
import yaml
from caf.toolkit import ToolDetails, LogHelper
from caf.brain.machine_vision.machine_vision_inputs import ObjectDetectionSatellitesInputs
from caf.brain.machine_vision.object_detection_satellites.object_detection_main import main


def model_setup():
    """
    Function to set up logging and parameter inputs.

    Returns
    -------
    None
    """
    with open("run_config.yaml", "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    params = ObjectDetectionSatellitesInputs(**config_data)

    output_path = os.path.join(params.output_path, "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = os.path.join(output_path, "log_file.log")
    details = ToolDetails("caf.brAIn Generate satellite images", "1.0.0")

    with LogHelper("caf.brain", details, console=True, log_file=path):
        main(params, output_path)


if __name__ == "__main__":
    model_setup()

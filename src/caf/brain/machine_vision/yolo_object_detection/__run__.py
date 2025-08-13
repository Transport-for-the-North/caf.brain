# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 5/14/2025
Original author: Adil Zaheer
"""
# Built-Ins
import os

# Third Party
import yaml
from caf.toolkit import LogHelper, ToolDetails

# Local Imports
from caf.brain.machine_vision.__inputs__ import ObjectDetectionInputs
from caf.brain.machine_vision.yolo_object_detection.__main__ import main


def model_setup():
    with open("run_config.yaml", "r") as file:
        config_data = yaml.safe_load(file)

    params = ObjectDetectionInputs(**config_data)

    output_path = os.path.join(params.output, "ObjectDetectionOutputs")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = os.path.join(output_path, "log_file.log")
    details = ToolDetails("caf.brAIn Object Detection", "1.0.0")

    with LogHelper("caf.brain", details, console=True, log_file=path):
        main(params)


if __name__ == "__main__":
    model_setup()

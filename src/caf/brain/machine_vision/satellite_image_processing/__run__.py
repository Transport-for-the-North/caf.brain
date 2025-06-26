# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 3/3/2025
Original author: Adil Zaheer
"""
import os
import yaml
from caf.toolkit import ToolDetails, LogHelper
from caf.brain.machine_vision.__inputs__ import GenerateSatelliteImagesInput
from caf.brain.machine_vision.satellite_image_processing.__main__ import main

def model_setup():
    with open('generate_satellite_images_config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

    params = GenerateSatelliteImagesInput(**config_data)

    output_path = os.path.join(params.output_path, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = os.path.join(output_path, 'log_file.log')
    details = ToolDetails("caf.brAIn Generate satellite images", "1.0.0")

    with LogHelper("caf.brain", details, console=True, log_file=path):
        main(params)


if __name__ == "__main__":
    model_setup()

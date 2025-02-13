# -*- coding: utf-8 -*-
"""
Created on: 1/15/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.MODELS.prediction_model.__main__ import main
from caf.ml.MODELS.prediction_model.prediction_model_inputs import run_file_inputs, Models
import yaml
from caf.toolkit import LogHelper, ToolDetails
import os


def model_setup():
    with open('prediction_config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

    if isinstance(config_data['model_choice'], str):
        config_data['model_choice'] = [Models[config_data['model_choice']]]
    else:
        config_data['model_choice'] = [Models[model] for model in config_data['model_choice']]

    params = run_file_inputs(**config_data)

    output_path = os.path.join(params.output_path, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = os.path.join(output_path, 'log_file.log')
    details = ToolDetails("caf.brAIn Prediction Model", "1.0.0")

    with LogHelper("caf.ml", details, console=True, log_file=path):
        main(params)


if __name__ == '__main__':
    model_setup()

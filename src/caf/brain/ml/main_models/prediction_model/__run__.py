"""
Created on: 1/15/2025
Original author: Adil Zaheer
"""
import os
import yaml
from caf.toolkit import LogHelper, ToolDetails
from caf.brain.ml.main_models.prediction_model.__main__ import main
from caf.brain.ml.main_models.prediction_model.prediction_model_inputs import (
    PredictionModelInputs,
    Models,
)


def model_setup():
    """
    Function to set up logging files, output folders and input data
    for the caf.brAIn prediction model config run.
    """

    with open("empty_config.yaml", "r", encoding='UTF-8') as file:
        config_data = yaml.safe_load(file)

    params = PredictionModelInputs(**config_data)

    if isinstance(params.modelling.model_choice, str):
        params.modelling.model_choice = [Models[params.modelling.model_choice]]
    else:
        params.modelling.model_choice = [Models[model] for model in params.modelling.model_choice]

    output_path = os.path.join(params.paths.output_path, "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = os.path.join(output_path, "log_file.log")
    details = ToolDetails("caf.brAIn Prediction Model", "1.0.0")

    with LogHelper("caf.brain", details, console=True, log_file=path):
        main(params)


if __name__ == "__main__":
    model_setup()

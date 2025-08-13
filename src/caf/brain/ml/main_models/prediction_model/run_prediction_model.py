"""
Created on: 1/15/2025
Original author: Adil Zaheer
"""

# Built-Ins
import os
import pathlib

# Third Party
import yaml
from caf.toolkit import LogHelper, ToolDetails

# Local Imports
from caf.brain.ml import PredictionModelInputs
from caf.brain.ml import Models
from caf.brain.ml.main_models.prediction_model.__main__ import main


def model_setup():
    """
    Function to set up logging files, output folders and input data
    for the caf.brAIn prediction model config run.
    """

    yaml_path = pathlib.Path("src/caf/brain/ml/main_models/prediction_model/empty_config.yaml")
    with open(yaml_path, "r", encoding="UTF-8") as file:
        config_data = yaml.safe_load(file)

    params = PredictionModelInputs(**config_data)

    if isinstance(params.modelling.model_choice, str):
        params.modelling.model_choice = [Models[params.modelling.model_choice]]
    else:
        params.modelling.model_choice = [
            Models[model] for model in params.modelling.model_choice
        ]

    output_path = params.paths.output_path / "output"
    if not output_path.exists():
        os.makedirs(output_path)

    path = output_path / "log_file.log"
    details = ToolDetails("caf.brAIn Prediction Model", "1.0.0")

    with LogHelper("caf.brain", details, console=True, log_file=path):
        main(params, output_path)


if __name__ == "__main__":
    model_setup()

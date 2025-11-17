"""
Created on: 1/15/2025
Original author: Adil Zaheer
"""

# Built-Ins
import os
import argparse
from pathlib import Path

# Third Party
import yaml
from caf.toolkit import LogHelper, ToolDetails

# Local Imports
from caf.brain.ml._functions._ml_inputs import Models, PredictionModelInputs
from caf.brain.ml._functions.prediction_model_main import main


def load_yaml(config_path: Path) -> dict:
    if config_path is None:
        config_path = Path.cwd() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")

    with open(config_path, "r", encoding="UTF-8") as file:
        config_data = yaml.safe_load(file)

    return config_data


def model_setup():
    """
    Function to set up logging files, output folders and input data
    for the caf.brAIn prediction model config run.
    """
    parser = argparse.ArgumentParser(description="Run caf.brAIn prediction model.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to YAML config file that follows \
                        INSERT PATH TO GUIDANCE DOC ") # todo
    args = parser.parse_args()

    config_data = load_yaml(args.config_path)

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

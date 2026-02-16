"""
Created on: 15/1/2025
Original author: Adil Zaheer
"""

# Built-Ins
import argparse
import os
from pathlib import Path

# Third Party
from caf.toolkit import LogHelper, ToolDetails

# Local Imports
from caf.brain.ml._functions._ml_inputs import PredictionModelInputs
from caf.brain.ml._functions.prediction_model_main import main


def _custom_load_yaml(config_path: Path) -> PredictionModelInputs:
    """
    Loads the YAML configuration file for the caf.brAIn full machine learning
    pipline and returns its contents as a dictionary.

    If no path is provided, defaults to 'brain.yml' which you can put in the
    current working directory.

    Parameters
    ----------
    config_path: Path to the YAML configuration file.

    Returns
    -------
    Parsed contents of the YAML file as a dictionary.
    """
    if config_path is None:
        config_path = Path.cwd() / "brain.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")

    config_data = PredictionModelInputs.load_yaml(config_path)

    return config_data


def model_setup():
    """
    Function to set up logging files, output folders and input data
    for the caf.brAIn full machine learning pipline.
    """
    parser = argparse.ArgumentParser(
        description="Run caf.brAIn prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("brain.yml"),
        help=(
            "Path to YAML config file. You should use docs/config.rst \n"
            "as guidance and examples/brain.yml as a template.\n"
        ),
    )
    args = parser.parse_args()
    params = _custom_load_yaml(args.config)

    output_path = params.paths.output_path / "output"
    if not output_path.is_dir():
        os.makedirs(output_path)

    path = output_path / "log_file.log"
    details = ToolDetails("caf.brAIn Prediction Model", "1.0.0")

    with LogHelper("caf.brain", details, console=True, log_file=path):
        main(params, output_path)


if __name__ == "__main__":
    model_setup()

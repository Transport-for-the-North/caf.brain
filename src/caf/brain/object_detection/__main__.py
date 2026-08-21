"""
CLI / main entry point to run object detection inference with a trained YOLO
model.
"""

# Built-Ins
import argparse
import os
from pathlib import Path

# Third Party
from caf.toolkit import LogHelper, ToolDetails

# Local Imports
from caf.brain.object_detection import _object_detection


def _custom_load_yaml(config_path: Path) -> _object_detection.ObjectDetectionInputs:
    """
    Loads the YAML configuration file for the caf.brAIn Object Detection Model.

    If no path is provided, defaults to 'object_detection.yml' which you can
    put in the current working directory.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    Parsed contents of the YAML file as a dictionary.
    """
    if config_path is None:
        config_path = Path.cwd() / "object_detection.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")

    config_data = _object_detection.ObjectDetectionInputs.load_yaml(config_path)

    return config_data


def model_setup():
    """
    Set up logging files, output folders and input data for the caf.brAIn
    object detection model.
    """
    parser = argparse.ArgumentParser(
        description="Run caf.brAIn object detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("object_detection.yml"),
        help=(
            "Path to YAML config file. You should use https://cafbrain.readthedocs.io/en/stable/usage/object_detection_config.html \n"
            "as guidance and examples/object_detection.yml as a template.\n"
        ),
    )
    args = parser.parse_args()
    params = _custom_load_yaml(args.config)

    output_path = params.object_detection.output_path / "caf.brAIn_object_detection"
    if not output_path.is_dir():
        os.makedirs(output_path)

    path = output_path / "log_file.log"
    details = ToolDetails("caf.brAIn Object Detection Model", "1.0.0")

    with LogHelper("caf.brain", details, console=True, log_file=path):
        _object_detection.main(params, output_path)


if __name__ == "__main__":
    model_setup()

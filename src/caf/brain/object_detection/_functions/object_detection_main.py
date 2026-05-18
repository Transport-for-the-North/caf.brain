"""
Created on: 13/03/2026
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import time
from pathlib import Path

# Local Imports
from caf.brain.object_detection import generate_satellite_images
from caf.brain.object_detection._functions._object_detection_inputs import (
    ObjectDetectionInputs,
)
from caf.brain.object_detection._functions.prediction.prediction_functions import (
    main_prediction,
)

LOG = logging.getLogger(__name__)


def main(
    params: ObjectDetectionInputs,
    output_path: Path | str,
) -> None:
    """
    Main object detection prediction function.

    Use this function to predict on satellite images if a model path is provided.
    If no model path is provided, it is assumed that you are conducting
    junction detection with TfNs saved model B:/TfN_object_detection/best.pt.

    Parameters
    ----------
    params:
        Config file inputs
    output_path:
        Path to output file location. Should be generated during model setup
        if not passed directly.

    Returns
    -------
    None
    """
    LOG.info("Building object detection model.")

    if params.object_detection.model_path is None:
        raise ValueError(
            "Model path must be provided. To generate this, either \n"
            "run the train_object_detection_model or if doing junction \n"
            "detection, use the TfN model here: B:/TfN_object_detection/best.pt"
        )

    output_path = Path(output_path)

    if params.object_detection.user_location is None:
        raise ValueError("user_location cannot be None.")

    if params.object_detection.image_folder_path is None:
        raise ValueError("image_folder_path cannot be None.")
    image_folder_path = Path(params.object_detection.image_folder_path)

    model_path = Path(params.object_detection.model_path)

    start_time = time.time()

    LOG.info(
        "Generate_satellite_images has to be run in order to generate images for prediction."
    )
    output_dir = generate_satellite_images(
        user_location=params.object_detection.user_location,
        output_path=output_path,
        image_folder_path=image_folder_path,
    )

    main_prediction(
        output=output_path,
        images_to_predict_path=output_dir,
        model_path=model_path,
    )

    end_time = time.time()
    LOG.info("Total run time: %.2f seconds", end_time - start_time)
    LOG.info("Finished model build")

"""
Created on: 7/22/2025
Original author: Adil Zaheer
"""

import os
import time
import logging
from pathlib import Path
from caf.brain.machine_vision.object_detection_satellites.prediction_and_evaluation.prediction_main import (
    main_prediction,
)
from caf.brain.machine_vision.object_detection_satellites.satellite_image_processing.image_generation_main import (
    main_image_generation,
)
from caf.brain.machine_vision.object_detection_satellites.yolo_modelling.model_build_main import (
    main_model_building,
)

LOG = logging.getLogger(__name__)


def main(params, output_path: Path):
    """
    Main pipeline for satellite image object detection.

    Parameters
    ----------
    params:
    output_path: Path to output folder location.

    Returns
    -------
    None
    """
    start_time = time.time()

    output = os.path.join(output_path, "output")
    if not os.path.exists(output):
        os.makedirs(output)

    if params.generate_training_images:
        LOG.info(
            "Image generation taking place. These are used to be labelled \
                  and then fed into the model for training"
        )

        main_image_generation(
            image_folder_path=params.image_folder_path,
            output_path=output,
            satellite_image_metadata=params.satellite_image_metadata,
            user_locations_path=params.user_locations_path,
            x_coordinate=params.x_coordinate,
            y_coordinate=params.y_coordinate,
        )

        LOG.info(
            "Images are generated and ready for labelling. It is recommended \
                  to use YoloLabel to label your images for model development"
        )
        end_time = time.time()
        LOG.info("Total run time: %s seconds", (end_time - start_time))
        return

    if params.build_model:
        main_model_building(
            output_path=output,
            image_folder_path=params.image_folder_path,
            class_names=params.class_names,
            hyperparameter_optimisation=params.hyperparameter_optimisation,
        )

        end_time = time.time()
        LOG.info("Total run time: %s seconds", (end_time - start_time))
        return

    LOG.info("Object detection model running")

    main_prediction(
        prediction_images_folder=params.prediction_images_folder,
        output=output,
        class_names=params.class_names,
    )

    LOG.info("Object detection model finished running")

    end_time = time.time()
    LOG.info("Total run time: %s seconds", (end_time - start_time))

    return

"""
Created on: 10/10/2025
Original author: Adil Zaheer
"""

import logging
from pathlib import Path
import time
from caf.brain.machine_vision.src_object_detection.model_building.build_model_data_functions import (
    main_ttv_creation,
    generate_folders,
    ensure_labels,
    build_config,
)
from caf.brain.machine_vision.src_object_detection.model_building.main_model_building import (
    main_model_build,
)
from caf.brain.machine_vision.src_object_detection.object_detection_inputs import (
    ObjectDetectionInputs,
)
from caf.brain.machine_vision.src_object_detection.image_creation.crop_user_satellite_images.main import (
    image_crop,
)
from caf.brain.machine_vision.src_object_detection.image_creation.generate_satellite_image_information.main import (
    image_info_generation,
)
from caf.brain.machine_vision.src_object_detection.image_creation.generate_user_satellite_images.main import (
    locate_user_coordinates,
)
from caf.brain.machine_vision.src_object_detection.prediction.prediction_functions import (
    main_prediction,
)

LOG = logging.getLogger(__name__)


def main(params: ObjectDetectionInputs, output_path: Path) -> None:
    """
    Main function for running the object detection pipeline.

    Parameters
    ----------
    params: ObjectDetectionInputs
    output_path: Path to output folder.

    Returns
    -------
    None
    """
    if params.image_generation_inputs.generate_images:
        satellite_image_metadata = image_info_generation(
            output_path=output_path,
            image_folder_path=params.image_generation_inputs.image_folder_path,
        )

        user_image_metadata = locate_user_coordinates(
            user_locations_csv_path=params.user_locations_csv_path,
            output_path=output_path,
            satellite_image_metadata=satellite_image_metadata,
            image_folder_path=params.image_generation_inputs.image_folder_path,
            x_coordinate=params.image_generation_inputs.x_coordinate,
            y_coordinate=params.image_generation_inputs.y_coordinate,
        )

        _ = image_crop(
            user_image_metadata=user_image_metadata,
            output_path=output_path,
            satellite_image_metadata=satellite_image_metadata,
        )

        return

    if params.build_model_inputs.build_model:
        LOG.info("Running model build")
        start_time = time.time()
        main_output_folder = Path(output_path) / "ModelBuildingOutputs"
        main_output_folder.mkdir(parents=True, exist_ok=True)

        train_path = main_output_folder / "train"
        test_path = main_output_folder / "test"
        val_path = main_output_folder / "val"

        if not train_path.exists():
            train, test, validation = main_ttv_creation(
                user_images_folder_path=params.build_model_inputs.user_images_folder_path,
                output_path=main_output_folder
            )

            generate_folders(dictionary=train, folder_name="train", output=main_output_folder)
            generate_folders(dictionary=test, folder_name="test", output=main_output_folder)
            generate_folders(dictionary=validation, folder_name="val", output=main_output_folder)

            ensure_labels(folder_path=train_path)
            ensure_labels(folder_path=test_path)
            ensure_labels(folder_path=val_path)

            build_config(
                output=main_output_folder,
                class_names=params.build_model_inputs.classification_names,
            )

        model_dir = main_output_folder / "model_results"
        final_model_path = model_dir / "best.pt"
        if not final_model_path.exists():
            main_model_build(
                output=main_output_folder,
                hyperparameter_optimisation=params.build_model_inputs.hyperparameter_optimisation,
            )

        end_time = time.time()
        LOG.info("Total run time: %.2f seconds", end_time - start_time)
        LOG.info("Finished model build")

        return

    if params.prediction_inputs.prediction_object_detection:
        # File structure:
        #     ObjectDetectionResults/
        #       satellite_image_metadata.csv
        #       user_coordinate_data.csv
        #       images_for_machine_vision/
        #         cropped images...
        #       prediction_results_folder/
        #         predictions/
        #           labels/
        #           prediction_summary.csv
        model_dir = Path(output_path) / "ObjectDetectionResults"
        model_dir.mkdir(parents=True, exist_ok=True)

        satellite_image_metadata = image_info_generation(
            output_path=model_dir,
            image_folder_path=params.image_generation_inputs.image_folder_path,
        )

        user_image_metadata = locate_user_coordinates(
            user_locations_csv_path=params.user_locations_csv_path,
            output_path=model_dir,
            satellite_image_metadata=satellite_image_metadata,
            image_folder_path=params.image_generation_inputs.image_folder_path,
            x_coordinate=params.image_generation_inputs.x_coordinate,
            y_coordinate=params.image_generation_inputs.y_coordinate,
        )

        output_dir = image_crop(
            user_image_metadata=user_image_metadata,
            output_path=model_dir,
            satellite_image_metadata=satellite_image_metadata,
        )

        main_prediction(
            output=model_dir,
            images_to_predict_path=output_dir,
            model_path=params.prediction_inputs.model_path,
        )

        return

"""
Created on: 5/7/2025
Original author: Adil Zaheer
"""

import os
from caf.brain.machine_vision.object_detection_satellites.yolo_modelling.data_processing.train_test_validate_generation import (
    BuildImagesTT,
    ensure_labels,
)
from caf.brain.machine_vision.object_detection_satellites.yolo_modelling.data_processing.build_yolo_config import (
    build_config,
)
from caf.brain.machine_vision.object_detection_satellites.yolo_modelling.model_creation.model_creation_main import (
    main_model_build,
)
from pathlib import Path


def main_model_building(
    output_path: Path,
    image_folder_path: Path,
    class_names: list[str],
    hyperparameter_optimisation: str,
):
    """

    Parameters
    ----------
    output_path
    image_folder_path
    class_names
    hyperparameter_optimisation

    Returns
    -------

    """
    main_output_folder = os.path.join(output_path, "ModelBuildingOutputs")
    os.makedirs(main_output_folder, exist_ok=True)

    train_folder_name = os.path.join(main_output_folder, "train")
    if not os.path.exists(train_folder_name):
        imageprocessor = BuildImagesTT(
            output=main_output_folder, image_dict=None, image_location=image_folder_path
        )

        train, test, validation = imageprocessor.run_fullclass()

        imageprocessor.generate_folders(dictionary=train, folder_name="train")
        imageprocessor.generate_folders(dictionary=test, folder_name="test")
        imageprocessor.generate_folders(dictionary=validation, folder_name="val")

        train_path = os.path.join(main_output_folder, "train")
        test_path = os.path.join(main_output_folder, "test")
        val_path = os.path.join(main_output_folder, "val")

        ensure_labels(folder_path=train_path)
        ensure_labels(folder_path=test_path)
        ensure_labels(folder_path=val_path)

    build_config(output=main_output_folder, class_names=class_names)

    model_dir = os.path.join(main_output_folder, "model_results")
    final_model_path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(final_model_path):
        main_model_build(
            output=main_output_folder, hyperparameter_optimisation=hyperparameter_optimisation
        )

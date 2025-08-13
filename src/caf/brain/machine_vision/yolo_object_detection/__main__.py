# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 5/7/2025
Original author: Adil Zaheer
"""
# Built-Ins
import os
import sys

# Local Imports
from caf.brain.machine_vision.yolo_object_detection.object_detection_pipeline.build_yolo_config import (
    build_config,
)
from caf.brain.machine_vision.yolo_object_detection.object_detection_pipeline.model_build_main import (
    main_model_build,
)
from caf.brain.machine_vision.yolo_object_detection.object_detection_pipeline.prediction_and_evaluation import (
    extract_results,
    prediction,
)
from caf.brain.machine_vision.yolo_object_detection.object_detection_pipeline.train_test_validate_generation import (
    BuildImagesTT,
    ensure_labels,
)


def main(params):
    main_output_folder = os.path.join(params.output, "ObjectDetectionOutputs")
    os.makedirs(main_output_folder, exist_ok=True)

    train_folder_name = os.path.join(main_output_folder, "train")
    if not os.path.exists(train_folder_name):
        imageprocessor = BuildImagesTT(
            output=main_output_folder, image_dict=None, image_location=params.image_path
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

    build_config(output=main_output_folder, class_names=params.class_names)

    model_dir = os.path.join(main_output_folder, "model_results")
    final_model_path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(final_model_path):
        final_model_path = main_model_build(
            output=main_output_folder,
            hyperparameter_optimisation=params.hyperparameter_optimisation,
            path_to_code=params.path_to_code,
        )

    prediction_results_dir, model = prediction(
        output=main_output_folder, final_model_path=final_model_path
    )

    extract_results(prediction_results_dir=prediction_results_dir, model=model)

    # todo apply comprehensive evaluation, can use what ive already written? IoU etc?

    return

# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 5/8/2025
Original author: Adil Zaheer
"""
import os
from caf.brain.machine_vision.yolo_object_detection.object_detection_pipeline.yolo_models import baseline_model, final_model
from caf.brain.machine_vision.yolo_object_detection.hyperparameter_optimisation_yolo.hyperparameter_optimisation_main import main_hyperparameter_optimisation


def main_model_build(output, hyperparameter_optimisation, path_to_code):
    model_dir = os.path.join(output, 'model_results')
    os.makedirs(model_dir, exist_ok=True)

    config_path = os.path.join(output, 'config.yaml')

    model = baseline_model(output=model_dir, config_path=config_path)

    main_hyperparameter_optimisation(basemodel=model,
                                     config=config_path,
                                     hyperparameter_optimisation=hyperparameter_optimisation,
                                     output=model_dir,
                                     path_to_code=path_to_code)

    final_training_results, final_model_path = final_model(output=model_dir,
                                                           config_path=config_path)

    return final_model_path

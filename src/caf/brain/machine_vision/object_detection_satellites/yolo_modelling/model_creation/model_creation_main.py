"""
Created on: 5/8/2025
Original author: Adil Zaheer
"""

import os
from caf.brain.machine_vision.object_detection_satellites.yolo_modelling.model_creation.model_build import (
    baseline_model,
    final_model,
)
from caf.brain.machine_vision.object_detection_satellites.yolo_modelling.hyperparameter_optimisation_yolo.hyperparameter_optimisation_main import (
    main_hyperparameter_optimisation,
)


def main_model_build(output, hyperparameter_optimisation):
    model_dir = os.path.join(output, "model_results")
    os.makedirs(model_dir, exist_ok=True)

    config_path = os.path.join(output, "config.yaml")

    model = baseline_model(output=model_dir, config_path=config_path)

    main_hyperparameter_optimisation(
        basemodel=model,
        config=config_path,
        hyperparameter_optimisation=hyperparameter_optimisation,
        output=model_dir,
    )

    final_model(output=model_dir, config_path=config_path)

    return

# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 5/12/2025
Original author: Adil Zaheer
"""
import os.path
import pandas as pd
import yaml
import time
import torch
from caf.brain.machine_vision.backlog.exhaustive_hyp_optim import exhaustive_optimisation
from caf.brain.machine_vision.yolo_object_detection.hyperparameter_optimisation_yolo.moderate_hyp_optim import moderate_optimisation
import logging
LOG = logging.getLogger(__name__)


def main_hyperparameter_optimisation(basemodel, config, hyperparameter_optimisation, output, path_to_code):
    """
    Hyperparameter_dict (for moderate and exhaustive methods):
    {'best_hyperparameters': {# Optimised hyperparameters (lr0, lrf, weight_decay, etc.)
                              # Exact keys may vary slightly between methods},
    'performance_metrics': {'mAP50': float or None,        # mean average precision at IoU 0.50
                            'mAP50-95': float or None,     # mean average precision across IoU thresholds
                            'precision': float or None,    # model precision
                            'recall': float or None,       # model recall},
    'trial_details': {'trial_id': str or int,        # identifier for the best trial
                      'last_result': float or dict   # performance of the best trial}}

    else case yaml output (standard YOLO format):
    lr0: 0.00269
    lrf: 0.00288
    momentum: 0.73375
    weight_decay: 0.00015
    etc...

    The best hyperparameters for all methods are saved to 'your_output_directory/best_hyperparameters.yaml'.
    They are saved in a flat yaml file ready for the final model.
    """
    start_time = time.time()

    hyperparameter_dir = os.path.join(output, 'hyperparameter_results')
    os.makedirs(hyperparameter_dir, exist_ok=True)

    if hyperparameter_optimisation == 'moderate' or None:
        LOG.info("Moderate hyperparamter optimisation is running")
        hyperparameter_dict = moderate_optimisation(config=config, output_dir=hyperparameter_dir, model=basemodel, path_to_code=path_to_code)

    else:
        LOG.info("Simple hyperparamter optimisation is running")
        torch.set_num_threads(8)
        hyperparameter_dict = basemodel.tune(config,
                                             project=hyperparameter_dir,
                                             epochs=30,
                                             iterations=300,
                                             imgsz=640,
                                             workers=8,
                                             optimizer="AdamW",
                                             plots=True,
                                             save=True,
                                             val=True,
                                             use_ray=True)

        custom_yaml_path = os.path.join(hyperparameter_dir, "best_hyperparameters.yaml")
        with open(custom_yaml_path, 'w') as f:
            yaml.dump(hyperparameter_dict, f)

    df_flat = pd.json_normalize(hyperparameter_dict)
    df_flat.to_csv(os.path.join(hyperparameter_dir, 'best_hyperparameters.csv'), index=False)

    end_time = time.time()
    LOG.info(f"Total hyperparamter optimisation run time: {end_time - start_time:.2f} seconds")
    LOG.info("Hyperparamter Optimisation running")

    return

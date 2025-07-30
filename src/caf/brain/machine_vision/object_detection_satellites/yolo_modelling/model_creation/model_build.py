"""
Created on: 5/9/2025
Original author: Adil Zaheer
"""
import os
import pandas as pd
from ultralytics import YOLO
import yaml
import shutil
import torch
import time
import logging
LOG = logging.getLogger(__name__)


def baseline_model(output, config_path):
    LOG.info("Base model running")
    torch.set_num_threads(8)

    model_dir = os.path.join(output, 'baseline_model_results')
    os.makedirs(model_dir, exist_ok=True)

    best_weights = os.path.join(model_dir, 'baseline', 'weights', 'best.pt')

    if os.path.exists(best_weights):
        print(f"Loading existing model from {best_weights}")
        model = YOLO(best_weights)
    else:
        model = YOLO('../yolo11n.pt')
        model.train(data=config_path,
                    epochs=50,
                    imgsz=640,
                    batch=16,
                    patience=10,
                    project=model_dir,
                    name='baseline',
                    exist_ok=True,
                    device=0)

    metrics = YOLO(best_weights).val(data=config_path,
                                     device=0)

    base_results = {'model_path': best_weights,
                    'map50': metrics.box.map50,
                    'map50_95': metrics.box.map,
                    'precision': metrics.box.mp,
                    'recall': metrics.box.mr}

    results_path = os.path.join(model_dir, 'base_model_results.csv')
    if not os.path.exists(results_path):
        df = pd.DataFrame([base_results])
        df.to_csv(results_path, index=False)

    LOG.info("Base model finished")
    return model


def final_model(output, config_path):
    LOG.info("Final model running")
    start_time = time.time()

    torch.set_num_threads(8)

    hyp_path = os.path.join(output, 'hyperparameter_results', 'best_hyperparameters.yaml')
    print(hyp_path)
    if os.path.exists(hyp_path):
        with open(os.path.join(hyp_path), 'r') as f:
            best_hyperparams = yaml.safe_load(f)
    else:
        LOG.error("best_hyperparameters does not exist. Please provide them or run \
                          hyperparamter optimisation functions")
        raise ValueError("best_hyperparameters does not exist. Please provide them or run \
                          hyperparamter optimisation functions")

    learning_params = best_hyperparams.copy()
    setup_params_to_remove = ['data',
                              'epochs',
                              'batch',
                              'imgsz',
                              'patience',
                              'project',
                              'name',
                              'exist_ok',
                              'workers']

    for param in setup_params_to_remove:
        if param in learning_params:
            removed_value = learning_params.pop(param)
            print(f"Removing {param}={removed_value} (will use explicit final training value)")

    final = YOLO('../yolo11n.pt')
    results = final.train(data=config_path,
                          epochs=150,
                          patience=20,
                          imgsz=640,
                          batch=16,
                          workers=8,
                          project=output,
                          name='final_model',
                          exist_ok=True,
                          device=0,
                          **learning_params)

    best_weights = os.path.join(output, 'final_model', 'weights', 'best.pt')
    final_model_path = os.path.join(output, 'best.pt')
    shutil.copy(best_weights, final_model_path)
    print(f"Final model saved to {final_model_path}")

    metrics = YOLO(best_weights).val(data=config_path,
                                     device=0)

    final_training_results = {'model_path': best_weights,
                              'map50': metrics.box.map50,
                              'map50_95': metrics.box.map,
                              'precision': metrics.box.mp,
                              'recall': metrics.box.mr}

    df = pd.DataFrame.from_dict([final_training_results])
    df.to_csv(os.path.join(output, 'final_training_results.csv'), index=False)

    end_time = time.time()
    LOG.info(f"Total final model run time: {end_time - start_time:.2f} seconds")
    LOG.info("Final model finished")

    return final_training_results, final_model_path

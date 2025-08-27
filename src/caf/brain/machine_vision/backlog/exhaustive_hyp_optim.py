# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 5/12/2025
Original author: Adil Zaheer
"""
# Built-Ins
import os

# Third Party
import ray
import yaml
from ray import tune
from ray.tune.search.optuna import OptunaSearch


def ray_optuna_objective_func(model, config_path, config):
    results = model.train(data=config_path, epochs=30, imgsz=640, batch=16, **config)

    metrics = model.val(data=config_path)

    results = {
        "metrics/mAP50-95(B)": metrics.box.map,
        "metrics/mAP50(B)": metrics.box.map50,
        "metrics/precision(B)": metrics.box.precision,
        "metrics/recall(B)": metrics.box.recall,
    }
    return results


def exhaustive_optimisation(model, config_path, output_dir):
    search_space = {
        "lr0": tune.loguniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "warmup_epochs": tune.uniform(0.0, 5.0),
        "cls": tune.uniform(0.2, 4.0),
        "hsv_h": tune.uniform(0.0, 0.1),
        "hsv_s": tune.uniform(0.0, 0.9),
        "hsv_v": tune.uniform(0.0, 0.9),
        "degrees": tune.uniform(0.0, 45.0),
        "scale": tune.uniform(0.1, 0.9),
        "shear": tune.uniform(0.0, 10.0),
        "perspective": tune.uniform(0.0, 0.001),
        "flipud": tune.uniform(0.0, 1.0),
        "mosaic": tune.uniform(0.0, 1.0),
    }

    optuna_search = OptunaSearch(metric="metrics/mAP50-95(B)", mode="max")

    ray_functionality_dir = os.path.join(output_dir, "ray_functionality_folder")
    os.makedirs(ray_functionality_dir, exist_ok=True)
    os.environ["TEMP_DIR"] = ray_functionality_dir
    ray.init(_temp_dir=ray_functionality_dir)

    ray_results_dir = os.path.join(output_dir, "hyperparameter_optim_ray_files")
    os.makedirs(ray_results_dir, exist_ok=True)

    analysis = tune.run(
        tune.with_parameters(ray_optuna_objective_func, config_path=config_path, model=model),
        storage_path=ray_results_dir,
        name="exhaustive_hyp_optim",
        search_alg=optuna_search,
        config=search_space,
        num_samples=20,
        resources_per_trial={"cpu": 8},
        metric="metrics/mAP50-95(B)",
        mode="max",
        trial_name_creator=lambda trial: f"trial_{trial.trial_id}",
    )

    best_trial = analysis.best_trial
    best_config = analysis.best_config
    best_result = analysis.best_result

    optimisation_results = {
        "best_hyperparameters": best_config,
        "performance_metrics": {
            "mAP50": best_result.get("metrics/mAP50(B)", None),
            "mAP50-95": best_result.get("metrics/mAP50-95(B)", None),
            "precision": best_result.get("metrics/precision(B)", None),
            "recall": best_result.get("metrics/recall(B)", None),
        },
        "trial_details": {
            "trial_id": best_trial.trial_id,
            "last_result": best_trial.last_result,
        },
    }

    yaml_path = os.path.join(output_dir, "best_hyperparameters.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(optimisation_results["best_hyperparameters"], f)

    return optimisation_results

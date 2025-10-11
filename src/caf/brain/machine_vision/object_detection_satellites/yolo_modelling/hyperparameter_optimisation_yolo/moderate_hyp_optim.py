"""
Created on: 5/12/2025
Original author: Adil Zaheer
"""

import os
import yaml
import optuna
from ultralytics import YOLO
import torch
import logging

LOG = logging.getLogger(__name__)


def moderate_optimisation(config, output_dir, model):
    best_hyp_path = os.path.join(output_dir, "best_hyperparameters.yaml")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=10, interval_steps=2
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="moderate_hyp_optim",
        storage="sqlite:///moderate_hyp_optim.db",
        load_if_exists=True,
    )

    def track_progress(study, trial):
        LOG.info(f"Trial {trial.number} finished with value: {trial.value}")
        LOG.info(f"Best value so far: {study.best_value}")

    if os.path.exists(best_hyp_path):
        LOG.info(f"Study already completed. Skipping optimisation.")
        LOG.info(f"Best trial: {study.best_trial.number}, Best value: {study.best_value}")
    else:
        LOG.info(f"Conducting optimisation")
        study.optimize(
            lambda trial: optuna_objective_func(trial, config, output_dir, model),
            n_trials=100,
            callbacks=[track_progress],
            show_progress_bar=True,
        )

    best_weights = os.path.join(output_dir, "train", "weights", "best.pt")
    best_weight_pt = torch.load(best_weights)
    hypparams = best_weight_pt["train_args"]
    with open(best_hyp_path, "w") as f:
        yaml.dump(hypparams, f)

    if os.path.exists(best_weights):
        try:
            metrics = YOLO(best_weights).val(data=config)
            performance_metrics = {
                "mAP50": metrics.box.map50,
                "mAP50-95": metrics.box.map,
                "precision": metrics.box.mp,
                "recall": metrics.box.mr,
            }
            LOG.info(f"Successfully extracted metrics: {performance_metrics}")
        except Exception as e:
            LOG.error(f"Error extracting metrics: {e}")
            performance_metrics = {
                "mAP50": study.best_value,
                "mAP50-95": None,
                "precision": None,
                "recall": None,
            }
    else:
        LOG.warning(f"Best weights not found at {best_weights}")
        performance_metrics = {
            "mAP50": study.best_value,
            "mAP50-95": None,
            "precision": None,
            "recall": None,
        }

    optimisation_results = {
        "best_hyperparameters": study.best_params,
        "performance_metrics": performance_metrics,
        "trial_details": {
            "trial_id": study.best_trial.number,
            "last_result": study.best_trial.value,
        },
    }

    return optimisation_results


# def cleanup_runs(path_to_code):
#     runs_path = os.path.join(path_to_code, 'machine_vision', 'yolo_modelling', 'runs')
#     if os.path.exists(runs_path):
#         LOG.info(f"Removing YOLO runs directory: {runs_path}")
#         os.remove(runs_path)


def optuna_objective_func(trial, config, output, model):
    # torch.set_num_threads(8)

    # search_space = {"lr0": trial.suggest_float("lr0", 1e-5, 1e-1, log=True),
    #                 "lrf": trial.suggest_float("lrf", 0.01, 1.0),
    #                 "momentum": trial.suggest_float("momentum", 0.6, 0.98),
    #                 "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),  # was 0.0, 0.001
    #                 "warmup_epochs": trial.suggest_float("warmup_epochs", 0.0, 5.0),
    #
    #                 "cls": trial.suggest_float("cls", 0.2, 4.0),
    #                 "box": trial.suggest_float("box", 0.05, 0.7),
    #
    #                 "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.1),
    #                 "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.9),
    #                 "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.9),
    #                 "degrees": trial.suggest_float("degrees", 0.0, 45.0),
    #                 "scale": trial.suggest_float("scale", 0.1, 0.9),  # was 0.0
    #                 "shear": trial.suggest_float("shear", 0.0, 10.0),
    #                 "perspective": trial.suggest_float("perspective", 0.0, 0.001),
    #                 "flipud": trial.suggest_float("flipud", 0.0, 1.0),
    #                 "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
    #                 "mixup": trial.suggest_float("mixup", 0.0, 1.0)}

    reduced_search_space = {
        "lr0": trial.suggest_float("lr0", 0.005, 0.02),
        "lrf": trial.suggest_float("lrf", 0.1, 0.3),
        "momentum": trial.suggest_float("momentum", 0.8, 0.95),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 3),
        "mosaic": trial.suggest_float("mosaic", 0.5, 1.0),
        "scale": trial.suggest_float("scale", 0.3, 0.7),
    }

    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

    try:
        results = model.train(
            data=config,
            epochs=30,
            imgsz=640,
            batch=16,
            patience=7,
            workers=8,
            project=output,
            name="",
            exist_ok=True,
            optimizer=optimizer,
            device=0,
            **reduced_search_space,
        )

        best_weights = os.path.join(output, "train", "weights", "best.pt")
        if not os.path.exists(best_weights):
            LOG.warning(f"Best weights not found for trial {trial.number}")
            return 0.0

        metrics = YOLO(best_weights).val(data=config)
        map50_95 = metrics.box.map

        trial.report(map50_95, step=30)

        return map50_95

    except Exception as e:
        LOG.error(f"Error in trial {trial.number}: {str(e)}")
        return 0.0

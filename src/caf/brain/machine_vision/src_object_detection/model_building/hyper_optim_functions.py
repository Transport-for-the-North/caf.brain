"""
Created on: 03/11/2025
Original author: Adil Zaheer
"""

from pathlib import Path
import time
import os
import yaml
import optuna
from ultralytics import YOLO
import torch
import logging

LOG = logging.getLogger(__name__)


def _moderate_optimisation(config: str | Path, output_dir: str | Path, model: YOLO) -> dict:
    """
    Run moderate hyperparameter optimisation for YOLO object detection
    using Optuna.

    Optuna runs up to 100 trials with a reduced search space and a median pruner.
    The best hyperparameters are saved to YAML, and performance metrics are
    extracted from the best weights.

    Parameters
    ----------
    config: Path to the YOLO dataset configuration file (YAML).
    output_dir: Directory where optimisation results, weights, and
                hyperparameters are stored.
    model: A YOLO model instance to be trained during optimisation. Generated
           from _baseline_model.

    Returns
    -------
    Dictionary containing:
    - "best_hyperparameters": dict of the best parameter values.
    - "performance_metrics": dict of evaluation metrics (mAP, precision, recall).
    - "trial_details": dict with best trial ID and value
    """

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        LOG.info("GPU available and being used to run the model")
    else:
        LOG.warning("GPU not available. CPU being used.")
        torch.set_num_threads(8)

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

    def _track_progress(study, trial):
        LOG.info("Trial %s finished with value: %s", trial.number, trial.value)
        LOG.info("Best value so far: %s", study.best_value)

    if os.path.exists(best_hyp_path):
        LOG.info("Study already completed. Skipping optimisation.")
        LOG.info("Best trial: %s, Best value: %s", study.best_trial.number, study.best_value)
    else:
        LOG.info("Conducting optimisation")
        study.optimize(
            lambda trial: _optuna_objective_func(trial, config, output_dir, model, device),
            n_trials=100,
            callbacks=[_track_progress],
            show_progress_bar=True,
        )

    best_weights = os.path.join(output_dir, "train", "weights", "best.pt")
    best_weight_pt = torch.load(best_weights)
    hypparams = best_weight_pt["train_args"]
    with open(best_hyp_path, "w") as f:
        yaml.dump(hypparams, f)

    if os.path.exists(best_weights):
        try:
            metrics = YOLO(best_weights).val(data=config, device=device)
            performance_metrics = {
                "mAP50": metrics.box.map50,
                "mAP50-95": metrics.box.map,
                "precision": metrics.box.mp,
                "recall": metrics.box.mr,
            }
            LOG.info("Successfully extracted metrics: %s", performance_metrics)
        except (RuntimeError, OSError, ValueError) as e:
            LOG.error(f"Error extracting metrics: {e}")
            performance_metrics = {
                "mAP50": study.best_value,
                "mAP50-95": None,
                "precision": None,
                "recall": None,
            }
    else:
        LOG.warning("Best weights not found at %s", best_weights)
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


def _optuna_objective_func(
    trial: optuna.Trial,
    config: str | Path,
    output: str | Path,
    model: YOLO,
    device: float | str,
) -> float:
    """
    Objective function for Optuna hyperparameter optimisation.

    Defines a hyperparameter search space (learning rate, momentum,
    weight decay, warmup, mosaic, scale, optimiser) and trains the YOLO model
    for 30 epochs. After training, evaluates the model and returns the mAP50-95
    score as the optimisation target.

    Parameters
    ----------
    trial: The current Optuna trial object.
    config: Path to the YOLO dataset configuration file (YAML).
    output: Directory where trial outputs and weights are stored.
    model: A YOLO model instance to train.
    device: Signifies what hardware (GPU or CPU) to run the optimisation on.

    Returns
    -------
    The mAP50-95 score for the trained model.
    """

    reduced_search_space = {
        "lr0": trial.suggest_float("lr0", 0.005, 0.02),
        "lrf": trial.suggest_float("lrf", 0.1, 0.3),
        "momentum": trial.suggest_float("momentum", 0.8, 0.95),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 3),
        "mosaic": trial.suggest_float("mosaic", 0.5, 1.0),
        "scale": trial.suggest_float("scale", 0.3, 0.7),
    }

    optimiser = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

    try:
        _ = model.train(
            data=config,
            epochs=30,
            imgsz=640,
            batch=16,
            patience=7,
            workers=8,
            project=output,
            name="",
            exist_ok=True,
            optimizer=optimiser,
            device=device,
            **reduced_search_space,
        )

        best_weights = os.path.join(output, "train", "weights", "best.pt")
        if not os.path.exists(best_weights):
            LOG.warning("Best weights not found for trial %s", trial.number)
            return 0.0

        metrics = YOLO(best_weights).val(data=config, device=device)
        map50_95 = metrics.box.map

        trial.report(map50_95, step=30)

        return map50_95

    except Exception as e:  # pylint: disable=broad-exception-caught
        LOG.error("Error in trial %s: %s", trial.number, str(e))
        return 0.0

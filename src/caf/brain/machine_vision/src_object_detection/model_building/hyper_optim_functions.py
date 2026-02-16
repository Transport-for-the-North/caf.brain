"""
Created on: 03/11/2025
Original author: Adil Zaheer
"""

from pathlib import Path
import os
import yaml
import optuna
from ultralytics import YOLO
import torch
import logging
from optuna.study import Study
from optuna.trial import FrozenTrial

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
        with open(best_hyp_path, "w") as f:
            yaml.dump(study.best_params, f)

    optimisation_results = {
        "best_hyperparameters": study.best_params,
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
        "lr0": trial.suggest_float("lr0", 0.008, 0.012),
        "lrf": trial.suggest_float("lrf", 0.15, 0.25),
        "momentum": trial.suggest_float("momentum", 0.88, 0.93),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-5, log=True),
        "mosaic": trial.suggest_float("mosaic", 0.7, 0.9),
        "scale": trial.suggest_float("scale", 0.4, 0.6),

    }
    optimiser = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

    try:
        _ = model.train(
            data=config,
            epochs=100,
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


class EarlyStopper:
    """
    Optuna early stopping callback for hyperparameter optimisation.

    This monitors the fitness score (trial.value) across iterations.
    If no improvement is observed for a set number of consecutive trials
    (patience), the study is stopped.

    Parameters
    ----------
    patience: Number of consecutive trials without improvement before stopping.

    Returns
    -------
    None
    """
    def __init__(self, patience: int = 5) -> None:
        self.patience: int = patience
        self.best_score: float | None = None
        self.counter: int = 0

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """
        Called automatically after each trial. Updates best_score, counter
        and stops the study if patience is exceeded.

        Parameters
        ----------
        study: The Optuna study object managing the optimisation.
        trial: The completed trial containing its fitness value.

        Returns
        -------
        None
        """
        if self.best_score is None or trial.value > self.best_score:
            self.best_score = trial.value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("Early stopping: no improvement in %s trials.", self.patience)
            study.stop()

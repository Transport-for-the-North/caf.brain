"""Functions to build a YOLO object detection model"""

# Built-Ins
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Literal

# Third Party
import pandas as pd
import strictyaml
import torch
from ultralytics import YOLO

# Local Imports
from caf.brain.object_detection.model_building import hyperparameters

LOG = logging.getLogger(__name__)


def _baseline_model(output: Path, config_path: Path) -> YOLO:
    """
    Base YOLO model run.

    Algorithm (no hyperparameter optimisation) is trained with the given
    config file. After training, the model is validated and key metrics are
    saved to CSV.

    Parameters
    ----------
    output:
        Directory where model results and weights will be stored.
    config_path:
        Path to the YOLO dataset configuration file (YAML).

    Returns
    -------
    The trained YOLO model instance.
    """
    LOG.info("Base model running")
    start_time = time.time()

    model_dir = output / "baseline_model_results"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_weights = model_dir / "baseline" / "weights" / "best.pt"
    device = select_device()

    if os.path.exists(best_weights):
        LOG.info("Loading existing model from %s", best_weights)
        model = YOLO(best_weights)
    else:
        model = YOLO("yolo11l.pt")
        model.train(
            data=config_path,
            epochs=300,
            imgsz=640,
            batch=16,
            patience=20,
            workers=8,
            project=model_dir,
            name="baseline",
            exist_ok=True,
            device=device,
        )

    metrics = YOLO(best_weights).val(data=config_path, device=device)

    base_results = {
        "model_path": best_weights,
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }

    results_path = os.path.join(model_dir, "base_model_results.csv")
    if not os.path.exists(results_path):
        df = pd.DataFrame([base_results])
        df.to_csv(results_path, index=False)

    end_time = time.time()
    LOG.info("Total baseline model run time: %.2f seconds", end_time - start_time)
    LOG.info("Base model finished")
    return model


def _final_model(output: Path, config_path: Path) -> None:
    """
    Train and evaluate the final YOLO object detection model.

    Uses the best hyperparameters from prior optimisation and the baseline
    model setup. If a GPU is available, training runs on GPU; otherwise it
    falls back to CPU. Saves the best weights, evaluation metrics, and a
    CSV summary to the output directory.

    Parameters
    ----------
    output:
        Directory where model results and weights will be stored.
    config_path:
        Path to the YOLO dataset configuration file (YAML).

    Returns
    -------
    None
    """
    LOG.info("Final model running")

    model_dir = output / "final_model_results"
    model_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    device = select_device()

    hyp_path = os.path.join(
        output, "hyperparameter_results", "tune", "best_hyperparameters.yaml"
    )
    if os.path.exists(hyp_path):
        with open(hyp_path, "r", encoding="utf-8") as f:
            best_hyperparams = strictyaml.load(f.read()).data
    else:
        raise ValueError(
            "best_hyperparameters does not exist. Please provide"
            " them or run hyperparameter optimisation functions"
        )

    learning_params = best_hyperparams.copy()
    for param in [
        "data",
        "epochs",
        "batch",
        "imgsz",
        "patience",
        "project",
        "name",
        "exist_ok",
        "workers",
    ]:
        learning_params.pop(param, None)

    learning_params = clean_hyperparams(learning_params)

    final = YOLO("yolo11l.pt")
    _ = final.train(
        data=config_path,
        epochs=300,
        patience=20,
        imgsz=640,
        batch=16,
        workers=8,
        project=model_dir,
        name="final_model",
        exist_ok=True,
        device=device,
        **learning_params,
    )

    best_weights = model_dir / "final_model" / "weights" / "best.pt"

    metrics = YOLO(best_weights).val(data=config_path, device=device)

    final_training_results = {
        "model_path": best_weights,
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }

    results_path = os.path.join(model_dir, "final_model_results.csv")
    if not os.path.exists(results_path):
        df = pd.DataFrame([final_training_results])
        df.to_csv(results_path, index=False)

    end_time = time.time()
    LOG.info("Total final model run time: %.2f seconds", end_time - start_time)
    LOG.info("Final model finished")


def _model_comparison(output: Path) -> None:
    """
    Runs a comparison between the baseline and final models.

    Parameters
    ----------
    output:
        Directory where model results from baseline and fine are stored.

    Returns
    -------
    None
    """
    baseline_csv = Path(output) / "baseline_model_results" / "base_model_results.csv"
    final_csv = Path(output) / "final_model_results" / "final_model_results.csv"

    baseline = pd.read_csv(baseline_csv).iloc[0]
    final = pd.read_csv(final_csv).iloc[0]

    if final["map50_95"] >= baseline["map50_95"]:
        LOG.info("Final model outperforms baseline")
        best_model_path = Path(final["model_path"])
        best_metrics = final
    else:
        LOG.info("Baseline model outperforms final")
        best_model_path = Path(baseline["model_path"])
        best_metrics = baseline

    optimal_dir = Path(output) / "optimal_model"
    optimal_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, optimal_dir / "best.pt")
    pd.DataFrame([best_metrics]).to_csv(optimal_dir / "optimal_model_results.csv", index=False)


def clean_hyperparams(hyp_dict: dict) -> dict:
    """
    Cleans hyperparameters inside the hyperparameter dictionary.

    Parameters
    ----------
    hyp_dict:
        Input hyperparameter dictionary.

    Returns
    -------
    Cleaned hyperparameter dictionary.
    """
    sanitised = {}
    for k, v in hyp_dict.items():
        if k == "close_mosaic":
            sanitised[k] = int(v)
        elif k in {"epochs", "batch", "workers", "patience"}:
            sanitised[k] = int(v)
        else:
            sanitised[k] = v
    return sanitised


def select_device() -> int | str:
    """
    Select the compute device (GPU or CPU).

    If a CUDA GPU is available, the function returns `0` and logs that the GPU
    will be used. If no GPU is available, it returns `cpu`, logs a warning,
    and reduces the number of CPU threads for better performance.

    Returns
    -------
    Int for GPU, cpu string for cpu.
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        LOG.info("GPU available and being used to run the model")
    else:
        LOG.warning("GPU not available. CPU being used.")
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 1
        torch.set_num_threads(cpu_count - 1)

    return device


def main_model_build(
    output: Path, hyperparameter_optimisation: Literal["base", "moderate"] | None = "base"
) -> None:
    """
    End-to-end pipeline to build a YOLO object detection model.

    Steps:
    1. Train a baseline model.
    2. Run hyperparameter optimisation (moderate Optuna or YOLO's tuner).
    3. Train the final model with the best hyperparameters.

    Parameters
    ----------
    output:
        Root directory where model results, configs, and outputs will be stored.
    hyperparameter_optimisation:
        Optimisation mode to pass to "main_hyperparameter_optimisation".
        "moderate" uses Optuna and "base" or None uses YOLO's built-in tuner.

    Returns
    -------
    None
    """
    model_dir = Path(output) / "model_results"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_dir / "best.pt"
    if final_model_path.exists():
        LOG.info(
            "Trained model already exists here: %s meaning you have already ran the model",
            final_model_path,
        )
    else:
        config_path = Path(output) / "config.yaml"

        model = _baseline_model(output=model_dir, config_path=config_path)

        hyperparameters.main_hyperparameter_optimisation(
            basemodel=model,
            config=config_path,
            hyperparameter_optimisation=hyperparameter_optimisation,
            output=model_dir,
        )

        _final_model(output=model_dir, config_path=config_path)
        _model_comparison(output=model_dir)

"""
Created on: 31/10/2025
Original author: Adil Zaheer
"""

import time
import logging
import shutil
import os
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import yaml
import torch


LOG = logging.getLogger(__name__)


def _baseline_model(output: Path, config_path: Path) -> YOLO:
    """
    Base YOLO model (no hyperparameter optimisation) is trained with the given
    config file. After training, the model is validated and key metrics are
    saved to CSV.

    Parameters
    ----------
    output: Directory where model results and weights will be stored.
    config_path: Path to the YOLO dataset configuration file (YAML).

    Returns
    -------
    The trained YOLO model instance.
    """
    LOG.info("Base model running")
    start_time = time.time()

    model_dir = os.path.join(output, "baseline_model_results")
    os.makedirs(model_dir, exist_ok=True)

    best_weights = os.path.join(model_dir, "baseline", "weights", "best.pt")
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        LOG.info("GPU available and being used to run the model")
    else:
        LOG.warning("GPU not available. CPU being used.")
        torch.set_num_threads(8)

    if os.path.exists(best_weights):
        print(f"Loading existing model from {best_weights}")
        model = YOLO(best_weights)
    else:
        model = YOLO("../yolo11n.pt")
        model.train(
            data=config_path,
            epochs=50,
            imgsz=640,
            batch=16,
            patience=10,
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
    output: Directory where model results and weights will be stored.
    config_path: Path to the YOLO dataset configuration file (YAML).

    Returns
    -------
    None
    """
    LOG.info("Final model running")
    start_time = time.time()

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        LOG.info("GPU available and being used to run the model")
    else:
        LOG.warning("GPU not available. CPU being used.")
        torch.set_num_threads(8)

    hyp_path = os.path.join(output, "hyperparameter_results", "best_hyperparameters.yaml")
    if os.path.exists(hyp_path):
        with open(hyp_path, "r", encoding="utf-8") as f:
            best_hyperparams = yaml.safe_load(f)
    else:
        LOG.error(
            "best_hyperparameters does not exist. Please provide them or run \
                          hyperparameter optimisation functions"
        )
        raise ValueError(
            "best_hyperparameters does not exist. Please provide them or run \
                          hyperparameter optimisation functions"
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

    final = YOLO("../yolo11n.pt")
    _ = final.train(
        data=config_path,
        epochs=150,
        patience=20,
        imgsz=640,
        batch=16,
        workers=8,
        project=output,
        name="final_model",
        exist_ok=True,
        device=device,
        **learning_params,
    )

    best_weights = os.path.join(output, "final_model", "weights", "best.pt")
    final_model_path = os.path.join(output, "best.pt")
    shutil.copy(best_weights, final_model_path)
    print(f"Final model saved to {final_model_path}")

    metrics = YOLO(best_weights).val(data=config_path, device=device)

    final_training_results = {
        "model_path": best_weights,
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }

    pd.DataFrame([final_training_results]).to_csv(
        os.path.join(output, "final_training_results.csv"), index=False
    )

    end_time = time.time()
    LOG.info("Total final model run time: %.2f seconds", end_time - start_time)
    LOG.info("Final model finished")

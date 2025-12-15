"""
Created on: 03/11/2025
Original author: Adil Zaheer
"""

from pathlib import Path
import os
import time
import logging
import pandas as pd
import torch
from ultralytics import YOLO

from caf.brain.machine_vision.src_object_detection.model_building.build_model_functions import (
    _baseline_model,
    _final_model, _model_comparison,
)
from caf.brain.machine_vision.src_object_detection.model_building.hyper_optim_functions import (
    _moderate_optimisation, EarlyStopper
)

LOG = logging.getLogger(__name__)


def main_hyperparameter_optimisation(
    basemodel: YOLO,
    config: str | Path,
    hyperparameter_optimisation: str | None,
    output: str | Path,
) -> None:
    """
    Run hyperparameter optimisation for a YOLO model.

    Depending on the hyperparameter_optimisation argument, this will either:
    - Run a moderate Optuna-based optimisation.
    - Run YOLO's built-in `.tune()` method.

    The best hyperparameters are saved to
    <output>/hyperparameter_results/best_hyperparameters.yaml and also
    exported as a CSV for inspection.

    Parameters
    ----------
    basemodel: A YOLO model instance, from _baseline_model.
    config: Path to the YOLO dataset configuration file (YAML).
    hyperparameter_optimisation : Optimisation mode. "moderate" uses Optuna
    and "base" or None uses YOLO's built-in tuner.
    output: Directory where optimisation results will be stored.

    Returns
    -------
    None
    """
    start_time = time.time()

    hyperparameter_dir = os.path.join(output, "hyperparameter_results")
    os.makedirs(hyperparameter_dir, exist_ok=True)

    if hyperparameter_optimisation == "moderate":
        LOG.info("Moderate hyperparameter optimisation is running")
        hyperparameter_dict = _moderate_optimisation(
            config=config, output_dir=hyperparameter_dir, model=basemodel
        )

        df_flat = pd.DataFrame([hyperparameter_dict])
        df_flat.to_csv(os.path.join(hyperparameter_dir, "best_hyperparameters.csv"), index=False)

    elif hyperparameter_optimisation == "base" or hyperparameter_optimisation is None:
        LOG.info("Simple hyperparameter optimisation is running")
        device = 0 if torch.cuda.is_available() else "cpu"
        if device == 0:
            LOG.info("GPU available and being used to run the model")
        else:
            LOG.warning("GPU not available. CPU being used.")
            torch.set_num_threads(8)

        best_hyp_file = Path(hyperparameter_dir) / "tune" / "best_hyperparameters.yaml"
        if os.path.exists(best_hyp_file):
            LOG.info("Best hyperparameters already exist and are being loaded in")
        else:
            basemodel.callbacks["on_trial_end"] = EarlyStopper(patience=5) # stop if 5 trials plateau
            _ = basemodel.tune(
                data=config,
                project=hyperparameter_dir,
                epochs=200,
                iterations=25,
                imgsz=640,
                workers=8,
                optimizer="AdamW",
                plots=True,
                save=True,
                val=True,
                use_ray=False, # only one GPU so not needed
                device=device,
                patience=20,
                batch=16,
            )
            LOG.info("Hyperparameters written out to {C:USER_OUTPUT_PATH/output/ModelBuildingOutputs/model_results/hyperparameter_results/tune/best_hyperparameters.yaml}", )

    else:
        raise ValueError(
            "Unknown hyperparameter optimisation mode. Please \
                          choose from either base or moderate."
        )

    end_time = time.time()
    LOG.info("Total Hyperparameter optimisation run time: %.2f seconds", end_time - start_time)
    LOG.info("Hyperparameter optimisation finished")


def main_model_build(output: Path, hyperparameter_optimisation: str | None = "base") -> None:
    """
    End-to-end pipeline to build, optimise, and train the final YOLO model for
    object detection.

    Steps:
    1. Train a baseline model.
    2. Run hyperparameter optimisation (moderate Optuna or YOLO's tuner).
    3. Train the final model with the best hyperparameters.

    Parameters
    ----------
    output: Root directory where model results, configs, and outputs
            will be stored.
    hyperparameter_optimisation: Optimisation mode to pass to
                                 "main_hyperparameter_optimisation". "moderate"
                                 uses Optuna and "base" or None uses
                                 YOLO's built-in tuner.

    Returns
    -------
    None
    """
    model_dir = Path(output) / "model_results"
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(output) / "config.yaml"

    model = _baseline_model(output=model_dir, config_path=config_path)

    main_hyperparameter_optimisation(
        basemodel=model,
        config=config_path,
        hyperparameter_optimisation=hyperparameter_optimisation,
        output=model_dir,
    )

    _final_model(output=model_dir, config_path=config_path)

    _model_comparison(output=model_dir)

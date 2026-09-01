"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import time
from pathlib import Path

# Third Party
import joblib
import pandas as pd

# Local Imports
from caf.brain.ml._functions._ml_inputs import PredictionModelInputs
from caf.brain.ml._functions.data_analysis.main import main_evaluate_input_data
from caf.brain.ml._functions.feature_selection.main import main_feature_selection
from caf.brain.ml._functions.hparam_optimisation.main import (
    main_hyperparameter_optimisation,
)
from caf.brain.ml._functions.model_selection.functions import initialise_model
from caf.brain.ml._functions.model_selection.main import main_model_selection
from caf.brain.ml._functions.prediction.main import main_prediction
from caf.brain.ml._functions.process_data_functions.main import main_input_data
from caf.brain.ml._functions.process_data_functions.split_data_into_ttv import (
    simple_train_test_split,
)

LOG = logging.getLogger(__name__)


def main(params: PredictionModelInputs, output_path: Path) -> None:
    """
    The main function for the caf.brAIn full machine learning pipeline.

    The prediction model utilises machine learning libraries in order to
    generate predictions. It's designed to streamline the process and remove
    any barrier to entry thereby making machine learning modelling more
    accessible.

    Parameters
    ----------
    params:
        Config file inputs
    output_path:
        Path to output file location. Should be generated during model setup
        if not passed directly.
    """
    start_time = time.time()

    paths = {
        "train_scaled": Path(output_path) / "train_scaled.csv",
        "test_scaled": Path(output_path) / "test_scaled.csv",
        "train_unscaled": Path(output_path) / "train_unscaled.csv",
        "test_unscaled": Path(output_path) / "test_unscaled.csv",
        "validate": Path(output_path) / "validate.csv",
    }

    if paths["train_scaled"].exists() and paths["test_scaled"].exists():
        LOG.info(
            "Training data is already present from a previous model run \n"
            "so it is being read in."
        )

        drop_vals_path = Path(output_path) / "dropped_encoding_vals.csv"
        if drop_vals_path.exists():
            drop_vals = pd.read_csv(drop_vals_path)
        else:
            drop_vals = None

        data = {
            name: pd.read_csv(path) if path.exists() else None for name, path in paths.items()
        }

        validate = data["validate"]

        custom_index = params.data_classification.custom_index
        if custom_index:
            for key in ["train_scaled", "test_scaled", "train_unscaled", "test_unscaled"]:
                df = data[key]
                if df is not None and all(col in df.columns for col in custom_index):
                    data[key] = df.set_index(custom_index, verify_integrity=False)

            if validate is not None and all(col in validate.columns for col in custom_index):
                validate = validate.set_index(custom_index, verify_integrity=False)

        train_scaled = data["train_scaled"]
        test_scaled = data["test_scaled"]
        train_unscaled = data["train_unscaled"]
        test_unscaled = data["test_unscaled"]

        if params.data_classification.numerical_features:
            num_pipe_path = Path(output_path) / "numerical_pipeline.pkl"
            numerical_pipeline = joblib.load(num_pipe_path)
        else:
            numerical_pipeline = None

    else:
        LOG.info(
            "Training data is not present from a previous model run so it \n"
            "is being generated."
        )

        data_dict, drop_vals, numerical_pipeline = main_input_data(
            output_path=output_path,
            paths=params.paths,
            data_classification=params.data_classification,
            transforming_inputs=params.transforming_inputs,
            modelling=params.modelling,
        )

        train_scaled = pd.DataFrame.from_dict(data_dict["train_scaled"])
        test_scaled = pd.DataFrame.from_dict(data_dict["test_scaled"])
        train_unscaled = pd.DataFrame.from_dict(data_dict["train_unscaled"])
        test_unscaled = pd.DataFrame.from_dict(data_dict["test_unscaled"])

        train_scaled.to_csv(paths["train_scaled"], index=True)
        test_scaled.to_csv(paths["test_scaled"], index=True)
        train_unscaled.to_csv(paths["train_unscaled"], index=True)
        test_unscaled.to_csv(paths["test_unscaled"], index=True)

        validate = None
        if data_dict["validate"] is not None and len(data_dict["validate"]) > 0:
            validate = pd.DataFrame.from_dict(data_dict["validate"])
            validate.to_csv(paths["validate"], index=True)

    if (
        train_scaled is None
        or test_scaled is None
        or train_unscaled is None
        or test_unscaled is None
    ):
        raise ValueError(
            "Train/test scaled and unscaled data must not be None at this stage. \n"
            "Please re-run the model and ensure input config is correctly populated."
        )

    selected_model = main_model_selection(
        paths=params.paths,
        data_classification=params.data_classification,
        transforming_inputs=params.transforming_inputs,
        modelling=params.modelling,
        train=train_scaled,
        output=output_path,
    )

    x_train, x_test, y_train, y_test, x_train_weight = simple_train_test_split(
        df=train_scaled,
        target_column=params.data_classification.target_column,
        weight_column=params.data_classification.weight_column,
    )

    x_train_model_fit, residuals, mse = initialise_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_train_weight=x_train_weight,
        output_folder=output_path,
        model_initialised=selected_model,
        classification_prediction=params.transforming_inputs.classification_prediction,
    )

    train_transformed, test_transformed = main_evaluate_input_data(
        paths=params.paths,
        data_classification=params.data_classification,
        transforming_inputs=params.transforming_inputs,
        modelling=params.modelling,
        output_path=output_path,
        train_scaled=train_scaled,
        test_scaled=test_scaled,
        train_unscaled=train_unscaled,
        test_unscaled=test_unscaled,
        model_fit=x_train_model_fit,
        model_initialised=selected_model,
        residuals=residuals,
        x_test=x_test,
        y_test=y_test,
        numerical_pipeline=numerical_pipeline,
    )

    train_final, test_final, cols_dropped_by_feat_select = main_feature_selection(
        paths=params.paths,
        data_classification=params.data_classification,
        transforming_inputs=params.transforming_inputs,
        modelling=params.modelling,
        train=train_transformed,
        test=test_transformed,
        output=output_path,
        initialised_model=selected_model,
    )

    best_model = main_hyperparameter_optimisation(
        paths=params.paths,
        data_classification=params.data_classification,
        transforming_inputs=params.transforming_inputs,
        modelling=params.modelling,
        train=train_final,
        model_instance=selected_model,
        output_folder=output_path,
    )

    main_prediction(
        model=best_model,
        test=test_final,
        target_column=params.data_classification.target_column,
        output_folder=output_path,
        validation=validate,
        weight_column=params.data_classification.weight_column,
        classification_prediction=params.transforming_inputs.classification_prediction,
        mse=mse,
        drop_vals=drop_vals,
        cols_dropped_by_feat_select=cols_dropped_by_feat_select,
    )

    end_time = time.time()
    LOG.info("Total run time: %.2f seconds", (end_time - start_time))

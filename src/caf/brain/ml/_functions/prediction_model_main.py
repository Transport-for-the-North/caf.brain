"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import time
from pathlib import Path

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml._functions.data_analysis.main import main_evaluate_input_data

from caf.brain.ml._functions.feature_selection.main import main_feature_selection
from caf.brain.ml._functions.hparam_optimisation.main import main_hyperparameter_optimisation
from caf.brain.ml._functions.model_selection.functions import initialise_model
from caf.brain.ml._functions.model_selection.main import main_model_selection
from caf.brain.ml._functions.prediction.main import main_prediction
from caf.brain.ml._functions.process_data_functions.main import main_input_data
from caf.brain.ml._functions.process_data_functions.split_data_into_ttv import (
    simple_train_test_split,
)
from caf.brain.ml._functions._ml_inputs import PredictionModelInputs

LOG = logging.getLogger(__name__)


def main(params: PredictionModelInputs, output_path: Path) -> None:
    """
    The main function for the caf.brAIn prediction model.
    The prediction model utilises machine learning libraries in order to
    generate predictions. It's designed to streamline the process and remove
    any barrier to entry thereby making machine learning modelling more
    accessible.

    Parameters
    ----------
    params: config file inputs
    output_path: path to output file location. Should be generated during
                 model setup if not passed directly.

    """
    start_time = time.time()

    data_dict, drop_vals, numerical_pipeline = main_input_data(
        output_path=output_path,
        paths=params.paths,
        data_classification=params.data_classification,
        transforming_inputs=params.transforming_inputs,
    )

    train_scaled = pd.DataFrame.from_dict(data_dict["train_scaled"])
    test_scaled = pd.DataFrame.from_dict(data_dict["test_scaled"])
    train_unscaled = pd.DataFrame.from_dict(data_dict["train_unscaled"])
    test_unscaled = pd.DataFrame.from_dict(data_dict["test_unscaled"])

    train_scaled.to_csv(output_path / "train_scaled.csv", index=True)
    test_scaled.to_csv(output_path / "test_scaled.csv", index=True)

    validate = None
    if data_dict["validate"] is not None and len(data_dict["validate"]) > 0:
        validate = pd.DataFrame.from_dict(data_dict["validate"])

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

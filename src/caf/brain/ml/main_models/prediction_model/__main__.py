"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import os
import time

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml.data_analysis.data_analysis_main import main_evaluate_input_data
from caf.brain.ml.feature_selection.feature_selection_main import main_feature_selection
from caf.brain.ml.hyperparameter_optimisation.hyper_optim_main import (
    main_hyperparameter_optimisation,
)
from caf.brain.ml.main_models.prediction_model.inputs import (
    PredictionModelInputs,
)
from caf.brain.ml.model_selection.functions import initialise_model
from caf.brain.ml.model_selection.main import main_model_selection
from caf.brain.ml.prediction.main import main_prediction
from caf.brain.ml.process_data_functions.process_data_main import main_input_data
from caf.brain.ml.process_data_functions.split_data_into_ttv import (
    simple_train_test_split,
)
from caf.brain.ml.statsmodel_pipeline.statsmodel_main import main_stats_model

LOG = logging.getLogger(__name__)


def main(params: PredictionModelInputs,
         output_path):
    """
    Main function for caf.brAIn prediction model.

    Parameters
    ----------
    params: config file inputs
    output_path

    Returns
    -------

    """
    start_time = time.time()

    data_dict, drop_vals, numerical_pipeline = main_input_data(
        output_path=output_path,
        paths=params.paths,
        data_classification=params.data_classification ,
        transforming_inputs=params.transforming_inputs,
    )

    train_scaled = pd.DataFrame.from_dict(data_dict["train_scaled"])
    test_scaled = pd.DataFrame.from_dict(data_dict["test_scaled"])
    train_unscaled = pd.DataFrame.from_dict(data_dict["train_unscaled"])
    test_unscaled = pd.DataFrame.from_dict(data_dict["test_unscaled"])

    train_scaled.to_csv(os.path.join(output_path, "train_scaled.csv"), index=True)
    test_scaled.to_csv(os.path.join(output_path, "test_scaled.csv"), index=True)

    validate = None
    if data_dict["validate"] is not None and len(data_dict["validate"]) > 0:
        validate = pd.DataFrame.from_dict(data_dict["validate"])

    is_statsmodel = any(
        base.__module__.startswith("statsmodels")
        for base in params.modelling.model_choice.__class__.__mro__
    )
    if is_statsmodel:
        main_stats_model(
            model_choice=params.modelling.model_choice,
            train=train_scaled,
            target_column=params.data_classification.target_column,
            weight_column=params.data_classification.weight_column,
        )

    selected_model = main_model_selection(paths=params.paths,
                                          data_classification=params.data_classification,
                                          transforming_inputs=params.transforming_inputs,
                                          modelling=params.modelling,
                                          train=train_scaled,
                                          output=output_path)

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
        x_train=x_train,
        x_test=x_test,
        numerical_pipeline=numerical_pipeline)

    train_final, test_final, cols_dropped_by_feat_select = main_feature_selection(
        paths=params.paths,
        data_classification=params.data_classification,
        transforming_inputs=params.transforming_inputs,
        modelling=params.modelling,
        train=train_transformed,
        test=test_transformed,
        output=output_path
    )

    best_model = main_hyperparameter_optimisation(
        train_final=train_final,
        target_column=params.target_column,
        model_instance=selected_model,
        model_name=params.model_choice,
        classification_prediction=params.classification_prediction,
        cv=params.cv,
        weight_column=params.weight_column,
        output_folder=output_path,
        is_time_series=params.is_time_series,
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

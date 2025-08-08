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
from caf.brain.ml.main_models.prediction_model.prediction_model_inputs import (
    PredictionModelInputs,
)
from caf.brain.ml.model_selection.model_selection_main import main_model_selection
from caf.brain.ml.prediction.prediction_main import main_prediction
from caf.brain.ml.process_data_functions.process_data_main import main_input_data
from caf.brain.ml.statsmodel_pipeline.statsmodel_main import main_stats_model

LOG = logging.getLogger(__name__)


def main(params: PredictionModelInputs):
    """
    Main function for caf.brAIn prediction model.

    Parameters
    ----------
    params: config file inputs

    """
    start_time = time.time()

    output_path = os.path.join(params.paths.output_path, "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_dict, drop_vals, numerical_pipeline = main_input_data(
        output_path=output_path,
        file_path=params.paths.file_path,
        folder_path=params.paths.folder_path,
        target_column=params.data_classification.target_column,
        custom_index=params.data_classification.custom_index,
        column_name_to_drop_rows=params.transforming_inputs.column_name_to_drop_rows,
        value_in_row=params.transforming_inputs.value_in_row,
        weight_column=params.data_classification.weight_column,
        categorical_features=params.data_classification.categorical_features,
        numerical_features=params.data_classification.numerical_features,
        classification_prediction=params.transforming_inputs.classification_prediction,
        split_by_value=params.transforming_inputs.split_by_value,
        validation_path=params.paths.validation_path,
        split_size=params.transforming_inputs.split_size,
        sample_size_encode=params.transforming_inputs.sample_size_encode,
        select_encode_values=params.transforming_inputs.select_encode_values,
        encode_values_to_drop=params.transforming_inputs.encode_values_to_drop,
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
        for base in params.model_choice.__class__.__mro__
    )
    if is_statsmodel:
        main_stats_model(
            model_choice=params.modelling.model_choice,
            train=train_scaled,
            target_column=params.data_classification.target_column,
            weight_column=params.data_classification.weight_column,
        )

    (model_initialised, x_train_model_fit, residuals, x_test, x_train, mse) = (
        main_model_selection(
            train=train_scaled,
            target_column=params.data_classification.target_column,
            weight_column=params.data_classification.weight_column,
            output=output_path,
            model=params.modelling.model_choice,
            classification_prediction=params.transforming_inputs.classification_prediction,
        )
    )

    train_transformed, test_transformed = main_evaluate_input_data(
        model_fit=x_train_model_fit,
        model_initialised=model_initialised,
        residuals=residuals,
        x_test=x_test,
        train_scaled=train_scaled,
        full_transformations=params.modelling.full_transformations,
        train_unscaled=train_unscaled,
        test_unscaled=test_unscaled,
        categorical_features=params.data_classification.categorical_features,
        numerical_features=params.data_classification.numerical_features,
        target_column=params.data_classification.target_column,
        weight_column=params.data_classification.weight_column,
        test_scaled=test_scaled,
        x_train=x_train,
        output_folder=output_path,
        is_time_series=params.data_classification.is_time_series,
        numerical_pipeline=numerical_pipeline,
    )

    train_final, test_final, cols_dropped_by_feat_select = main_feature_selection(
        train=train_transformed,
        test=test_transformed,
        target_column=params.data_classification.target_column,
        cv=params.modelling.cv,
        regression_method=model_initialised,
        weight_column=params.data_classification.weight_column,
        classification_prediction=params.transforming_inputs.classification_prediction,
        output=output_path,
        skip_feature_selection=params.modelling.skip_feature_selection,
        intensive_feature_selection=params.modelling.intensive_feature_selection,
        is_time_series=params.data_classification.is_time_series,
    )

    best_model = main_hyperparameter_optimisation(
        train_final=train_final,
        target_column=params.target_column,
        model_instance=model_initialised,
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

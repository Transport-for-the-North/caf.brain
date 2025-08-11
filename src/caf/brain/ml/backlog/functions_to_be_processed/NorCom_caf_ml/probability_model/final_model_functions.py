# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 11/20/2024
Original author: Adil Zaheer
"""
# Built-Ins
import os.path
import time

# Third Party
import joblib
from caf.ml.backlog.functions_to_be_processed.NorCom_caf_ml.probability_model.functions import (
    final_prediction,
    model_prep,
    modified_hyper_optimisation,
    refined_cafml_data_analysis,
    simple_eval_model,
)


def refined_cafml_model(
    training_df,
    test_df,
    validation_df,
    target_column,
    output_folder,
    weight_column,
    model_to_use,
    index_columns,
    binary_prediction,
):
    start_time = time.time()
    print("Refined cafml version running")

    model, residuals = model_prep(
        training_df=training_df,
        target_column=target_column,
        output_folder=output_folder,
        weight_column=weight_column,
        model_to_use=model_to_use,
    )

    refined_cafml_data_analysis(
        data=training_df,
        target_column=target_column,
        weight_column=weight_column,
        residuals=residuals,
    )

    final_model_filename = os.path.join(output_folder, "cafml_final_model.pkl")
    if os.path.exists(final_model_filename):
        print("loading final model")
        final_model = joblib.load(final_model_filename)
    else:
        final_model = modified_hyper_optimisation(
            model=model,
            data=training_df,
            target_column=target_column,
            output_folder=output_folder,
            weight_column=weight_column,
            original_training_data=training_df,
            index_columns=index_columns,
        )

    y_pred = final_prediction(
        model=final_model,
        data=test_df,
        target_column=target_column,
        output_folder=output_folder,
        validation=validation_df,
        binary_prediction=binary_prediction,
    )

    simple_eval_model(
        training_df=training_df,
        validation_df=validation_df,
        y_pred=y_pred,
        model=model,
        target_column=target_column,
        output_folder=output_folder,
    )

    end_time = time.time()
    print(f"Total run time: {end_time - start_time:.2f} seconds")

    return

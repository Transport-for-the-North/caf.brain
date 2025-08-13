# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 10/10/2024
Original author: Adil Zaheer
"""
# Third Party
from caf.ml.backlog.functions_to_be_processed.forecast_model_functions import (
    align_dataframes,
    predict_refined,
    process_data_loaded_model,
    process_forecast_data,
)
from caf.ml.backlog.functions_to_be_processed.model_algorithm_evaluation import (
    eval_model,
)
from caf.ml.backlog.functions_to_be_processed.NorCom_caf_ml.larger_cafml_model.norcom_specific_functions import (
    apply_transformations_norcom,
)


def saved_model_func(
    params,
    regression_method,
    hyperparameters,
    trained_data,
    transformations,
    data_analysis_data,
):
    trained_data = process_data_loaded_model(
        df=trained_data,
        index_columns=params.index_columns,
        drop_columns=params.drop_columns,
        target_column=params.target_column,
        keep_columns=params.keep_columns,
        outlier_threshold=params.outlier_threshold,
        categorical_target=params.categorical_target,
    )

    predict_data = process_forecast_data(
        df=params.predict_data,
        index_columns_p=params.index_columns_predict,
        drop_columns_p=params.drop_columns_predict,
        target_column=params.target_column,
        keep_columns_p=params.keep_columns_predict,
        outlier_threshold_p=params.outlier_threshold_predict,
        categorical_target=params.categorical_target,
        output_folder=params.output_folder,
    )

    dat = apply_transformations_norcom(
        predict_data=predict_data,
        transformations=transformations,
        target_column=params.target_column,
        numerical_features=params.numerical_features,
        categorical_features=params.categorical_features,
        output_folder=params.output_folder,
        training_data=trained_data,
        features_to_transform=None,
        training_data_pre_feat_selection=data_analysis_data,
    )

    final_predict_data = align_dataframes(
        df1=trained_data, df2=dat, output_folder=params.output_folder
    )

    forecasted_data, y_proba = predict_refined(
        trained_data=trained_data,
        predict_data=final_predict_data,
        trained_model=regression_method,
        target_column=params.target_column,
        output_folder=params.output_folder,
        FinalModelParameters=hyperparameters,
        index_col=params.index_columns,
    )

    print(forecasted_data)

    eval_model(
        data_used_to_predict=final_predict_data,
        data_contains_truth_values_only=params.validation_data,
        model_predicted_data=forecasted_data,
        model=regression_method,
        target_column=params.target_column,
        output_folder=params.output_folder,
        categorical_target=params.categorical_target,
        y_proba=y_proba,
    )

    return

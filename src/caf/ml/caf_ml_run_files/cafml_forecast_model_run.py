# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import os
import warnings
from caf.ml.functions_all import process_data_pipeline
from caf.ml.functions_all import eval_model
from caf.ml.functions_all import norcom_run_functions
from caf.ml.old_inputs.cafml_inputs import CarAccessInputs

warnings.filterwarnings("ignore")

from caf.ml.functions_all import DataProcessor
from caf.ml.functions_all import feature_selection_cv, filter_data
from caf.ml.functions_all import select_param
from caf.ml.functions_all import predict_refined, process_data_loaded_model, process_forecast_data, align_dataframes
from caf.ml.functions_all import save_model_and_parameters, load_model_and_parameters
from caf.ml.functions_all import pre_forecast_data_analysis, apply_transformations
# todo add interpolation function from lvu to dataprocessor class


def main(params: CarAccessInputs):
    if params.saved_model:
        return process_saved_model(params)
    else:
        return process_new_model(params)


def process_saved_model(params):
    forecasted_data = saved_model_functions(params)
    return forecasted_data


def saved_model_functions(params):
    regression_method, hyperparameters, trained_data, transformations = load_model_and_parameters(
        params.output_folder)

    trained_data = process_data_loaded_model(df=trained_data,
                                             index_columns=params.index_columns,
                                             drop_columns=params.drop_columns,
                                             target_column=params.target_column,
                                             keep_columns=params.keep_columns,
                                             outlier_threshold=params.outlier_threshold,
                                             categorical_target=params.categorical_target)

    predict_data = process_forecast_data(df=params.predict_data,
                                         index_columns_p=params.index_columns_predict,
                                         drop_columns_p=params.drop_columns_predict,
                                         target_column=params.target_column,
                                         keep_columns_p=params.keep_columns_predict,
                                         outlier_threshold_p=params.outlier_threshold_predict,
                                         categorical_target=params.categorical_target,
                                         output_folder=params.output_folder)

    predict_data_post_transformation = apply_transformations(predict_data=predict_data,
                                                             transformations=transformations,
                                                             target_column=params.target_column,
                                                             numerical_features=params.numerical_features,
                                                             categorical_features=params.categorical_features,
                                                             output_folder=params.output_folder,
                                                             training_data=trained_data)

    final_predict_data = align_dataframes(df1=trained_data, df2=predict_data_post_transformation, output_folder=params.output_folder)


    forecasted_data, y_proba = predict_refined(trained_data=trained_data,
                                              predict_data=final_predict_data,
                                              trained_model=regression_method,
                                              target_column=params.target_column,
                                              output_folder=params.output_folder,
                                              FinalModelParameters=hyperparameters,
                                              index_col=params.index_columns)

    print(forecasted_data)

    eval_model(data_used_to_predict=final_predict_data,
               data_contains_truth_values_only=params.validation_data,
               model_predicted_data=forecasted_data,
               model=regression_method,
               target_column=params.target_column,
               output_folder=params.output_folder,
               categorical_target=params.categorical_target,
               y_proba=y_proba)

    return forecasted_data


def process_new_model(params):
    return process_prediction(params)


def process_prediction(params):
    processed_data = DataProcessor(params.x_path,
                                   params.y_path,
                                   params.folder_path,
                                   params.index_columns,
                                   params.drop_columns,
                                   params.keep_columns,
                                   params.target_column,
                                   params.output_folder,
                                   params.wide_format,
                                   params.variable_name,
                                   params.value_name,
                                   params.outlier_threshold,
                                   params.categorical_target,
                                   params.column_name_to_drop_rows,
                                   params.value_in_row).data

    preprocessed_df, transformations_ = process_data_pipeline(df=processed_data,
                                                              numerical_features=params.numerical_features,
                                                              categorical_features=params.categorical_features,
                                                              target_column=params.target_column,
                                                              output_folder=params.output_folder)

    if params.norcom_run is not None:
        return process_regular_model(params,
                                     preprocessed_df,
                                     transformations_,
                                     model=None,
                                     selected_features_model=None,
                                     selected_f_regressor=None,
                                     selected_rfe=None,
                                     selected_sfs=None,
                                     selected_mutual_info=None)
    else:
        (selected_features_model,
         selected_f_regressor,
         selected_rfe,
         selected_sfs,
         selected_mutual_info,
         model) = feature_selection_cv(data=preprocessed_df,
                                       model_type=params.model_type,
                                       cv_method=params.cv_method,
                                       splits=params.splits,
                                       repeats=params.repeats,
                                       target_column=params.target_column,
                                       output_folder=params.output_folder,
                                       skip_feature_selection=params.skip_feature_selection,
                                       basic_model=params.basic_model,
                                       multiple_year_prediction=params.multiple_year_prediction,
                                       categorical_data=params.categorical_data)

    if params.skip_feature_selection:
        return process_skip_feature_selection(params, preprocessed_df, transformations_, model)
    elif params.skip_data_analysis:
        return process_skip_data_analysis(params, preprocessed_df, transformations_, model,
                                          selected_features_model,
                                          selected_f_regressor,
                                          selected_rfe,
                                          selected_sfs,
                                          selected_mutual_info)
    elif params.skip_hyperparameter_optimisation:
        return process_skip_hyperparameter_optimisation(params, preprocessed_df, transformations_, model,
                                                       selected_features_model,
                                                       selected_f_regressor,
                                                       selected_rfe,
                                                       selected_sfs,
                                                       selected_mutual_info)
    elif params.basic_model:
        return process_basic_model(params, preprocessed_df, transformations_, model)
    else:
        return process_regular_model(params, preprocessed_df, transformations_, model,
                                     selected_features_model,
                                     selected_f_regressor,
                                     selected_rfe,
                                     selected_sfs,
                                     selected_mutual_info)


def process_skip_feature_selection(params, processed_data, transformations_, model):
    df_final_to_model, transformations = pre_forecast_data_analysis(data=processed_data,
                                                                    regression_method=model,
                                                                    target_column=params.target_column,
                                                                    threshold=params.threshold,
                                                                    threshold_corr=params.threshold_corr,
                                                                    output_folder=params.output_folder,
                                                                    index_col=params.index_columns,
                                                                    categorical_data=params.categorical_data,
                                                                    categorical_features=params.categorical_features,
                                                                    categorical_transformations=transformations_)

    ## HYPERPARAMETER OPTIMISATION ##
    hyperparameters = select_param(data=df_final_to_model,
                                   target_column=params.target_column,
                                   model=model,
                                   categorical_data=params.categorical_data,
                                   multiple_year_prediction=params.multiple_year_prediction)

    ## SAVE MODEL ##
    save_model_and_parameters(regression_method=model,
                              hyperparameters=hyperparameters,
                              transformations=transformations,
                              output_folder=params.output_folder,
                              skip_data_analysis=params.skip_data_analysis,
                              skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation)

    ## FINAL PREDICTION ##
    predict_data = process_forecast_data(df=params.predict_data,
                                         index_columns_p=params.index_columns_predict,
                                         drop_columns_p=params.drop_columns_predict,
                                         target_column=params.target_column,
                                         keep_columns_p=params.keep_columns_predict,
                                         outlier_threshold_p=params.outlier_threshold_predict,
                                         categorical_target=params.categorical_target,
                                         output_folder=params.output_folder)

    dat = apply_transformations(predict_data=predict_data,
                                transformations=transformations,
                                target_column=params.target_column,
                                numerical_features=params.numerical_features,
                                categorical_features=params.categorical_features,
                                output_folder=params.output_folder)

    final_predict_data = align_dataframes(df1=df_final_to_model, df2=dat)

    predict_data_path = os.path.join(params.output_folder, 'Final_prediction_data.csv')
    predict_data.to_csv(predict_data_path, index=True)
    print(f"Predict data post transformations saved to: {predict_data_path}")

    forecasted_data, y_proba = predict_refined(trained_data=df_final_to_model,
                                      predict_data=final_predict_data,
                                      trained_model=model,
                                      target_column=params.target_column,
                                      output_folder=params.output_folder,
                                      FinalModelParameters=hyperparameters,
                                      index_col=params.index_columns)

    print(forecasted_data)

    eval_model(data_used_to_predict=final_predict_data,
               data_contains_truth_values_only=params.validation_data,
               model_predicted_data=forecasted_data,
               model=regression_method,
               target_column=params.target_column,
               output_folder=params.output_folder,
               categorical_target=params.categorical_target,
               y_proba=y_proba)

    return forecasted_data


def process_skip_data_analysis(params, preprocessed_df, transformations_, model,
                               selected_features_model,
                               selected_f_regressor,
                               selected_rfe,
                               selected_sfs,
                               selected_mutual_info):


    data = filter_data(original_data=preprocessed_df,
                       best_features_model=selected_features_model,
                       best_features_f_regressor=selected_f_regressor,
                       best_features_rfe=selected_rfe,
                       output_folder=params.output_folder,
                       target_column=params.target_column,
                       best_features_sfs=selected_sfs,
                       best_features_mutual_info=selected_mutual_info,
                       model=model,
                       index_col=params.index_columns)

    ## HYPERPARAMETER OPTIMISATION ##
    hyperparameters = select_param(data=data,
                                   target_column=params.target_column,
                                   model=model,
                                   categorical_data=params.categorical_data,
                                   multiple_year_prediction=params.multiple_year_prediction)

    ## SAVE MODEL ##
    save_model_and_parameters(regression_method=model,
                              hyperparameters=hyperparameters,
                              transformations=transformations_,
                              output_folder=params.output_folder,
                              skip_data_analysis=params.skip_data_analysis,
                              skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation)

    ## FINAL PREDICTION ##
    predict_data = process_forecast_data(df=params.predict_data,
                                         index_columns_p=params.index_columns_predict,
                                         drop_columns_p=params.drop_columns_predict,
                                         target_column=params.target_column,
                                         keep_columns_p=params.keep_columns_predict,
                                         outlier_threshold_p=params.outlier_threshold_predict,
                                         categorical_target=params.categorical_target,
                                         output_folder=params.output_folder)

    dat = apply_transformations(predict_data=predict_data,
                                transformations=transformations_,
                                target_column=params.target_column,
                                numerical_features=params.numerical_features,
                                categorical_features=params.categorical_features,
                                output_folder=params.output_folder)

    final_predict_data = align_dataframes(df1=data, df2=dat)

    predict_data_path = os.path.join(params.output_folder, 'Final_prediction_data.csv')
    predict_data.to_csv(predict_data_path, index=True)
    print(f"Predict data post transformations saved to: {predict_data_path}")

    forecasted_data, y_proba = predict_refined(trained_data=df_final_to_model,
                                      predict_data=final_predict_data,
                                      trained_model=model,
                                      target_column=params.target_column,
                                      output_folder=params.output_folder,
                                      FinalModelParameters=hyperparameters,
                                      index_col=params.index_columns)

    print(forecasted_data)

    eval_model(data_used_to_predict=final_predict_data,
               data_contains_truth_values_only=params.validation_data,
               model_predicted_data=forecasted_data,
               model=regression_method,
               target_column=params.target_column,
               output_folder=params.output_folder,
               categorical_target=params.categorical_target,
               y_proba=y_proba)

    return forecasted_data


def process_skip_hyperparameter_optimisation(params, preprocessed_df, transformations_, model,
                                             selected_features_model,
                                             selected_f_regressor,
                                             selected_rfe,
                                             selected_sfs,
                                             selected_mutual_info):

    data = filter_data(original_data=preprocessed_df,
                       best_features_model=selected_features_model,
                       best_features_f_regressor=selected_f_regressor,
                       best_features_rfe=selected_rfe,
                       output_folder=params.output_folder,
                       target_column=params.target_column,
                       best_features_sfs=selected_sfs,
                       best_features_mutual_info=selected_mutual_info,
                       model=model,
                       index_col=params.index_columns)

    df_final_to_model, transformations = pre_forecast_data_analysis(data=data,
                                                                    regression_method=model,
                                                                    target_column=params.target_column,
                                                                    threshold=params.threshold,
                                                                    threshold_corr=params.threshold_corr,
                                                                    output_folder=params.output_folder,
                                                                    index_col=params.index_columns,
                                                                    categorical_data=params.categorical_data,
                                                                    categorical_features=params.categorical_features,
                                                                    categorical_transformations=transformations_)

    ## SAVE MODEL ##
    save_model_and_parameters(regression_method=model,
                              hyperparameters=None,
                              transformations=transformations,
                              output_folder=params.output_folder,
                              skip_data_analysis=params.skip_data_analysis,
                              skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation)

    ## FINAL PREDICTION ##
    predict_data = process_forecast_data(df=params.predict_data,
                                         index_columns_p=params.index_columns_predict,
                                         drop_columns_p=params.drop_columns_predict,
                                         target_column=params.target_column,
                                         keep_columns_p=params.keep_columns_predict,
                                         outlier_threshold_p=params.outlier_threshold_predict,
                                         categorical_target=params.categorical_target,
                                         output_folder=params.output_folder)

    dat = apply_transformations(predict_data=predict_data,
                                transformations=transformations,
                                target_column=params.target_column,
                                numerical_features=params.numerical_features,
                                categorical_features=params.categorical_features,
                                output_folder=params.output_folder)

    final_predict_data = align_dataframes(df1=df_final_to_model, df2=dat)

    predict_data_path = os.path.join(params.output_folder, 'Final_prediction_data.csv')
    predict_data.to_csv(predict_data_path, index=True)
    print(f"Predict data post transformations saved to: {predict_data_path}")

    forecasted_data, y_proba = predict_refined(trained_data=df_final_to_model,
                                      predict_data=final_predict_data,
                                      trained_model=model,
                                      target_column=params.target_column,
                                      output_folder=params.output_folder,
                                      FinalModelParameters=None,
                                      index_col=params.index_columns)

    print(forecasted_data)

    eval_model(data_used_to_predict=final_predict_data,
               data_contains_truth_values_only=params.validation_data,
               model_predicted_data=forecasted_data,
               model=regression_method,
               target_column=params.target_column,
               output_folder=params.output_folder,
               categorical_target=params.categorical_target,
               y_proba=y_proba)


    return forecasted_data


def process_basic_model(params, preprocessed_df, transformations_, model):
    ## SAVE MODEL ##
    save_model_and_parameters(regression_method=model,
                              hyperparameters=None,
                              transformations=transformations_,
                              output_folder=params.output_folder,
                              skip_data_analysis=params.skip_data_analysis,
                              skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation)

    ## FINAL PREDICTION ##
    predict_data = process_forecast_data(df=params.predict_data,
                                         index_columns_p=params.index_columns_predict,
                                         drop_columns_p=params.drop_columns_predict,
                                         target_column=params.target_column,
                                         keep_columns_p=params.keep_columns_predict,
                                         outlier_threshold_p=params.outlier_threshold_predict,
                                         categorical_target=params.categorical_target,
                                         output_folder=params.output_folder)
    print('Predict data:')
    print(predict_data)
    print(type(predict_data))

    dat = align_dataframes(df1=preprocessed_df, df2=predict_data)
    print(type(dat))

    forecasted_data, y_proba = predict_refined(trained_data=preprocessed_df,
                                      predict_data=dat,
                                      trained_model=model,
                                      target_column=params.target_column,
                                      output_folder=params.output_folder,
                                      FinalModelParameters=None,
                                      index_col=params.index_columns)


    print(forecasted_data)

    eval_model(data_used_to_predict=final_predict_data,
               data_contains_truth_values_only=params.validation_data,
               model_predicted_data=forecasted_data,
               model=regression_method,
               target_column=params.target_column,
               output_folder=params.output_folder,
               categorical_target=params.categorical_target,
               y_proba=y_proba)

    return forecasted_data


def process_regular_model(params, preprocessed_df, transformations_, model,
                          selected_features_model,
                          selected_f_regressor,
                          selected_rfe,
                          selected_sfs,
                          selected_mutual_info):

    if params.norcom_run is not None:
        return norcom_run_functions(params, preprocessed_df=preprocessed_df, transformations_=transformations_)

    data = filter_data(original_data=preprocessed_df,
                       best_features_model=selected_features_model,
                       best_features_f_regressor=selected_f_regressor,
                       best_features_rfe=selected_rfe,
                       output_folder=params.output_folder,
                       target_column=params.target_column,
                       best_features_sfs=selected_sfs,
                       best_features_mutual_info=selected_mutual_info,
                       model=model,
                       index_col=params.index_columns)

    df_final_to_model, transformations = pre_forecast_data_analysis(data=data,
                                                                    regression_method=model,
                                                                    target_column=params.target_column,
                                                                    threshold=params.threshold,
                                                                    threshold_corr=params.threshold_corr,
                                                                    output_folder=params.output_folder,
                                                                    index_col=params.index_columns,
                                                                    categorical_data=params.categorical_data,
                                                                    categorical_features=params.categorical_features,
                                                                    categorical_transformations=transformations_,
                                                                    features_to_transform=params.features_to_transform)

    ## HYPERPARAMETER OPTIMISATION ##
    hyperparameters = select_param(data=df_final_to_model,
                                   target_column=params.target_column,
                                   model=model,
                                   categorical_data=params.categorical_data,
                                   multiple_year_prediction=params.multiple_year_prediction)

    ## SAVE MODEL ##
    save_model_and_parameters(regression_method=model,
                              hyperparameters=hyperparameters,
                              transformations=transformations,
                              output_folder=params.output_folder,
                              skip_data_analysis=params.skip_data_analysis,
                              skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation)

    ## FINAL PREDICTION ##
    predict_data = process_forecast_data(df=params.predict_data,
                                         index_columns_p=params.index_columns_predict,
                                         drop_columns_p=params.drop_columns_predict,
                                         target_column=params.target_column,
                                         keep_columns_p=params.keep_columns_predict,
                                         outlier_threshold_p=params.outlier_threshold_predict,
                                         categorical_target=params.categorical_target,
                                         output_folder=params.output_folder)

    dat = apply_transformations(predict_data=predict_data,
                                transformations=transformations,
                                target_column=params.target_column,
                                numerical_features=params.numerical_features,
                                categorical_features=params.categorical_features,
                                output_folder=params.output_folder,
                                training_data=df_final_to_model,
                                features_to_transform=params.features_to_transform)

    final_predict_data = align_dataframes(df1=df_final_to_model, df2=dat, output_folder=params.output_folder)


    forecasted_data, y_proba = predict_refined(trained_data=df_final_to_model,
                                      predict_data=final_predict_data,
                                      trained_model=model,
                                      target_column=params.target_column,
                                      output_folder=params.output_folder,
                                      FinalModelParameters=hyperparameters,
                                      index_col=params.index_columns)


    print(forecasted_data)

    eval_model(data_used_to_predict=final_predict_data,
               data_contains_truth_values_only=params.validation_data,
               model_predicted_data=forecasted_data,
               model=model,
               target_column=params.target_column,
               output_folder=params.output_folder,
               categorical_target=params.categorical_target,
               y_proba=y_proba)

    return forecasted_data


warnings.filterwarnings("default")

#todo indexing still causing massive issues, speak to someone about it

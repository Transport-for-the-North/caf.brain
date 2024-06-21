# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import os

import numpy as np
import warnings

from caf.ml.functions.data_pipeline_functions import process_data_pipeline
from caf.ml.functions.model_algorithm_evaluation import eval_model

warnings.filterwarnings("ignore")

from caf.ml.inputs.cafml_inputs import CarAccessInputs
from caf.ml.functions.process_data_class import DataProcessor
from caf.ml.functions.feature_selection import feature_selection_cv, filter_data
from caf.ml.functions.hyper_optim_gridsearch import select_param
from caf.ml.functions.forecast_model_functions import predict, process_data_loaded_model, process_forecast_data, align_dataframes
from caf.ml.functions.save_model import save_model_and_parameters, load_model_and_parameters
from caf.ml.functions.pre_forecast_data_analysis import pre_forecast_data_analysis, apply_transformations
# todo add interpolation function from lvu to dataprocessor class


def main(params: CarAccessInputs):
    ######## MODEL AND DATA SAVED ########
    if params.saved_model is not None:
        if params.categorical_data is not None:
            if params.single_year_prediction is not None:

                if params.skip_feature_selection is not None:
                    return

                if params.skip_hyperparameter_optimisation is not None:
                    return

                if params.skip_data_analysis is not None:
                    ## LOAD MODEL AND DATA ##
                    regression_method, hyperparameters, trained_data, transformations = load_model_and_parameters(
                        params.output_folder)
                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
                    print(hyperparameters)
                    print(transformations)
                    print(regression_method)

                    trained_data = process_data_loaded_model(df=trained_data,
                                                             index_columns=params.index_columns,
                                                             drop_columns=params.drop_columns,
                                                             target_column=params.target_column,
                                                             keep_columns=params.keep_columns,
                                                             outlier_threshold=params.outlier_threshold)

                    predict_data = process_forecast_data(df=params.predict_data,
                                                         index_columns_p=params.index_columns_predict,
                                                         drop_columns_p=params.drop_columns_predict,
                                                         target_column=params.target_column,
                                                         keep_columns_p=params.keep_columns_predict,
                                                         outlier_threshold_p=params.outlier_threshold_predict)

                    print('Trained data:')
                    print(trained_data)
                    print(type(trained_data))
                    print('Predict data:')
                    print(predict_data)
                    print(type(predict_data))

                    final_predict_data = align_dataframes(df1=trained_data, df2=predict_data)

                    print(final_predict_data)

                    ## FINAL FORECAST ##
                    forecasted_data = predict(params.single_year_prediction,
                                              trained_data=trained_data,
                                              predict_data=final_predict_data,
                                              trained_model=regression_method,
                                              target_column=params.target_column,
                                              output_folder=params.output_folder,
                                              FinalModelParameters=hyperparameters,
                                              index_col=params.index_columns,
                                              skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                              basic_model=params.basic_model)
                    print(forecasted_data)

                    eval_model(data_used_to_predict=final_predict_data,
                               data_contains_truth_values_only=params.validation_data,
                               model_predicted_data=forecasted_data,
                               model=regression_method,
                               target_column=params.target_column,
                               output_folder=params.output_folder)

                    return

                if params.basic_model is not None:
                    ## LOAD MODEL AND DATA ##
                    regression_method, hyperparameters, trained_data, transformations = load_model_and_parameters(
                        params.output_folder)
                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
                    print(hyperparameters)
                    print(transformations)
                    print(regression_method)

                    trained_data = process_data_loaded_model(df=trained_data,
                                                             index_columns=params.index_columns,
                                                             drop_columns=params.drop_columns,
                                                             target_column=params.target_column,
                                                             keep_columns=params.keep_columns,
                                                             outlier_threshold=params.outlier_threshold)

                    predict_data = process_forecast_data(df=params.predict_data,
                                                         index_columns_p=params.index_columns_predict,
                                                         drop_columns_p=params.drop_columns_predict,
                                                         target_column=params.target_column,
                                                         keep_columns_p=params.keep_columns_predict,
                                                         outlier_threshold_p=params.outlier_threshold_predict)

                    print('Trained data:')
                    print(trained_data)
                    print(type(trained_data))
                    print('Predict data:')
                    print(predict_data)
                    print(type(predict_data))

                    final_predict_data = align_dataframes(df1=trained_data, df2=predict_data)

                    print(final_predict_data)

                    ## FINAL FORECAST ##
                    forecasted_data = predict(params.single_year_prediction,
                                              trained_data=trained_data,
                                              predict_data=final_predict_data,
                                              trained_model=regression_method,
                                              target_column=params.target_column,
                                              output_folder=params.output_folder,
                                              FinalModelParameters=hyperparameters,
                                              index_col=params.index_columns,
                                              skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                              basic_model=params.basic_model)
                    print(forecasted_data)

                    eval_model(data_used_to_predict=final_predict_data,
                               data_contains_truth_values_only=params.validation_data,
                               model_predicted_data=forecasted_data,
                               model=regression_method,
                               target_column=params.target_column,
                               output_folder=params.output_folder)

                    return


            elif params.multiple_year_prediction is not None:

                if params.skip_feature_selection is not None:
                    return

                if params.skip_hyperparameter_optimisation is not None:
                    return

                if params.skip_data_analysis is not None:
                    ## LOAD MODEL AND DATA ##
                    regression_method, hyperparameters, trained_data, transformations = load_model_and_parameters(
                        params.output_folder)
                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
                    print(hyperparameters)
                    print(transformations)
                    print(regression_method)

                    trained_data = process_data_loaded_model(df=trained_data,
                                                             index_columns=params.index_columns,
                                                             drop_columns=params.drop_columns,
                                                             target_column=params.target_column,
                                                             keep_columns=params.keep_columns,
                                                             outlier_threshold=params.outlier_threshold)

                    predict_data = process_forecast_data(df=params.predict_data,
                                                         index_columns_p=params.index_columns_predict,
                                                         drop_columns_p=params.drop_columns_predict,
                                                         target_column=params.target_column,
                                                         keep_columns_p=params.keep_columns_predict,
                                                         outlier_threshold_p=params.outlier_threshold_predict)

                    print('Trained data:')
                    print(trained_data)
                    print(type(trained_data))
                    print('Predict data:')
                    print(predict_data)
                    print(type(predict_data))

                    final_predict_data = align_dataframes(df1=trained_data, df2=predict_data)

                    print(final_predict_data)

                    ## FINAL FORECAST ##
                    forecasted_data = predict(params.single_year_prediction,
                                              trained_data=trained_data,
                                              predict_data=final_predict_data,
                                              trained_model=regression_method,
                                              target_column=params.target_column,
                                              output_folder=params.output_folder,
                                              FinalModelParameters=hyperparameters,
                                              index_col=params.index_columns,
                                              skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                              basic_model=params.basic_model)
                    print(forecasted_data)

                    eval_model(data_used_to_predict=final_predict_data,
                               data_contains_truth_values_only=params.validation_data,
                               model_predicted_data=forecasted_data,
                               model=regression_method,
                               target_column=params.target_column,
                               output_folder=params.output_folder)

                    return

                if params.basic_model is not None:
                    ## LOAD MODEL AND DATA ##
                    regression_method, hyperparameters, trained_data, transformations = load_model_and_parameters(
                        params.output_folder)
                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
                    print(hyperparameters)
                    print(transformations)
                    print(regression_method)

                    trained_data = process_data_loaded_model(df=trained_data,
                                                             index_columns=params.index_columns,
                                                             drop_columns=params.drop_columns,
                                                             target_column=params.target_column,
                                                             keep_columns=params.keep_columns,
                                                             outlier_threshold=params.outlier_threshold)

                    predict_data = process_forecast_data(df=params.predict_data,
                                                         index_columns_p=params.index_columns_predict,
                                                         drop_columns_p=params.drop_columns_predict,
                                                         target_column=params.target_column,
                                                         keep_columns_p=params.keep_columns_predict,
                                                         outlier_threshold_p=params.outlier_threshold_predict)

                    print('Trained data:')
                    print(trained_data)
                    print(type(trained_data))
                    print('Predict data:')
                    print(predict_data)
                    print(type(predict_data))

                    final_predict_data = align_dataframes(df1=trained_data, df2=predict_data)

                    print(final_predict_data)

                    ## FINAL FORECAST ##
                    forecasted_data = predict(params.single_year_prediction,
                                              trained_data=trained_data,
                                              predict_data=final_predict_data,
                                              trained_model=regression_method,
                                              target_column=params.target_column,
                                              output_folder=params.output_folder,
                                              FinalModelParameters=hyperparameters,
                                              index_col=params.index_columns,
                                              skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                              basic_model=params.basic_model)
                    print(forecasted_data)

                    eval_model(data_used_to_predict=final_predict_data,
                               data_contains_truth_values_only=params.validation_data,
                               model_predicted_data=forecasted_data,
                               model=regression_method,
                               target_column=params.target_column,
                               output_folder=params.output_folder)

                    return

                regression_method, hyperparameters, trained_data, transformations = load_model_and_parameters(params.output_folder)

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
                                                                         output_folder=params.output_folder)

                final_predict_data = align_dataframes(df1=trained_data, df2=predict_data_post_transformation)


                file_path = os.path.join(params.output_folder, 'Final_prediction_data.csv')
                if os.path.exists(file_path):
                    print(f"The file Final_prediction_data.csv already exists in {params.output_folder} and is being replaced.")
                final_predict_data_path = os.path.join(params.output_folder, 'Final_prediction_data.csv')
                final_predict_data.to_csv(final_predict_data_path, index=True)
                print(f"Final prediction data saved to: {final_predict_data_path}")


                forecasted_data = predict(params.single_year_prediction,
                                          trained_data=trained_data,
                                          predict_data=final_predict_data,
                                          trained_model=regression_method,
                                          target_column=params.target_column,
                                          output_folder=params.output_folder,
                                          FinalModelParameters=hyperparameters,
                                          index_col=params.index_columns,
                                          skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                          basic_model=params.basic_model,
                                          multiple_year_prediction=params.multiple_year_prediction,
                                          categorical_data=params.categorical_data)

                print(forecasted_data)

                eval_model(data_used_to_predict=final_predict_data,
                           data_contains_truth_values_only=params.validation_data,
                           model_predicted_data=forecasted_data,
                           model=regression_method,
                           target_column=params.target_column,
                           output_folder=params.output_folder,
                           categorical_target=params.categorical_target)


    ######## SIMPLE FEATURE SELECTION ########
    if params.saved_model is None:
        if params.single_year_prediction is not None:
            ## PROCESS DATA ###
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
                                           params.outlier_threshold).data

            print('Checking if data is numeric')
            all_numeric = processed_data.applymap(np.isreal).all().all()
            print(all_numeric)

            ## FEATURE SELECTION ##
            (selected_features_model,
             selected_f_regressor,
             selected_rfe,
             selected_sfs,
             selected_mutual_info,
             model) = feature_selection_cv(data=processed_data,
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

            #### SKIP FEATURE SELECTION - INCLUDE DATA TRANSFORMATIONS ####
            if params.skip_feature_selection is not None:
                df_final_to_model, transformations = pre_forecast_data_analysis(data=processed_data,
                                                                                regression_method=model,
                                                                                target_column=params.target_column,
                                                                                threshold=params.threshold,
                                                                                threshold_corr=params.threshold_corr,
                                                                                output_folder=params.output_folder,
                                                                                index_col=params.index_columns)

                ## HYPERPARAMETER OPTIMISATION ##
                hyperparameters = select_param(data=df_final_to_model,
                                               target_column=params.target_column,
                                               model=model)

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
                                                     outlier_threshold_p=params.outlier_threshold_predict)

                print('Predict data:')
                print(predict_data)
                print(type(predict_data))

                dat = align_dataframes(df1=df_final_to_model, df2=predict_data)
                print(type(dat))

                final_predict_data = apply_transformations(predict_data=dat,
                                                           transformations=transformations)

                print(final_predict_data)

                forecasted_data = predict(params.single_year_prediction,
                                          trained_data=df_final_to_model,
                                          predict_data=final_predict_data,
                                          trained_model=model,
                                          target_column=params.target_column,
                                          output_folder=params.output_folder,
                                          FinalModelParameters=hyperparameters,
                                          index_col=params.index_columns,
                                          skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                          basic_model=params.basic_model)

                print(forecasted_data)

                eval_model(data_used_to_predict=final_predict_data,
                           data_contains_truth_values_only=params.validation_data,
                           model_predicted_data=forecasted_data,
                           model=model,
                           target_column=params.target_column,
                           output_folder=params.output_folder)

                return

            #### SKIP DATA TRANSFORMATIONS - INCLUDE FEATURE SELECTION ####
            if params.skip_data_analysis is not None:
                data = filter_data(original_data=processed_data,
                                   best_features_model=selected_features_model,
                                   best_features_f_regressor=selected_f_regressor,
                                   best_features_rfe=selected_rfe,
                                   output_folder=params.output_folder,
                                   target_column=params.target_column,
                                   best_features_sfs=selected_sfs,
                                   best_features_mutual_info=selected_mutual_info,
                                   model=model)

                ## HYPERPARAMETER OPTIMISATION ##
                hyperparameters = select_param(data=data,
                                               target_column=params.target_column,
                                               model=model)

                ## SAVE MODEL ##
                save_model_and_parameters(regression_method=model,
                                          hyperparameters=hyperparameters,
                                          transformations=None,
                                          output_folder=params.output_folder,
                                          skip_data_analysis=params.skip_data_analysis,
                                          skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation)

                ## FINAL PREDICTION ##
                predict_data = process_forecast_data(df=params.predict_data,
                                                     index_columns_p=params.index_columns_predict,
                                                     drop_columns_p=params.drop_columns_predict,
                                                     target_column=params.target_column,
                                                     keep_columns_p=params.keep_columns_predict,
                                                     outlier_threshold_p=params.outlier_threshold_predict)

                print('Predict data:')
                print(predict_data)
                print(type(predict_data))

                dat = align_dataframes(df1=data, df2=predict_data)
                print(type(dat))

                forecasted_data = predict(params.single_year_prediction,
                                          trained_data=data,
                                          predict_data=dat,
                                          trained_model=model,
                                          target_column=params.target_column,
                                          output_folder=params.output_folder,
                                          FinalModelParameters=hyperparameters,
                                          index_col=params.index_columns,
                                          skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                          basic_model=params.basic_model)

                print(forecasted_data)

                eval_model(data_used_to_predict=dat,
                           data_contains_truth_values_only=params.validation_data,
                           model_predicted_data=forecasted_data,
                           model=model,
                           target_column=params.target_column,
                           output_folder=params.output_folder)

                return

            #### SKIP HYPERPARAMETER OPTIMISATION - INCLUDE DATA TRANSFORMATIONS AND FEATURE SELECTION ####
            if params.skip_hyperparameter_optimisation is not None:
                data = filter_data(original_data=processed_data,
                                   best_features_model=selected_features_model,
                                   best_features_f_regressor=selected_f_regressor,
                                   best_features_rfe=selected_rfe,
                                   output_folder=params.output_folder,
                                   target_column=params.target_column,
                                   best_features_sfs=selected_sfs,
                                   best_features_mutual_info=selected_mutual_info,
                                   model=model)

                ## FINAL DATA ANALYSIS ##
                df_final_to_model, transformations = pre_forecast_data_analysis(data=data,
                                                                                regression_method=model,
                                                                                target_column=params.target_column,
                                                                                threshold=params.threshold,
                                                                                threshold_corr=params.threshold_corr,
                                                                                output_folder=params.output_folder,
                                                                                index_col=params.index_columns)

                print("Final data to be modelled:")
                print(df_final_to_model)


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
                                                     outlier_threshold_p=params.outlier_threshold_predict)

                print('Predict data:')
                print(predict_data)
                print(type(predict_data))

                dat = align_dataframes(df1=df_final_to_model, df2=predict_data)
                print(type(dat))

                final_predict_data = apply_transformations(predict_data=dat,
                                                           transformations=transformations)

                print(final_predict_data)

                forecasted_data = predict(params.single_year_prediction,
                                          trained_data=df_final_to_model,
                                          predict_data=final_predict_data,
                                          trained_model=model,
                                          target_column=params.target_column,
                                          output_folder=params.output_folder,
                                          FinalModelParameters=None,
                                          index_col=params.index_columns,
                                          skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                          basic_model=params.basic_model)

                print(forecasted_data)

                eval_model(data_used_to_predict=final_predict_data,
                           data_contains_truth_values_only=params.validation_data,
                           model_predicted_data=forecasted_data,
                           model=model,
                           target_column=params.target_column,
                           output_folder=params.output_folder)


                return

            #### MODEL WITH NO EXTRAS - AS BASIC AS POSSIBLE ####
            if params.basic_model is not None:

                ## SAVE MODEL ##
                save_model_and_parameters(regression_method=model,
                                          hyperparameters=None,
                                          transformations=None,
                                          output_folder=params.output_folder,
                                          skip_data_analysis=params.skip_data_analysis,
                                          skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation)

                ## FINAL PREDICTION ##
                predict_data = process_forecast_data(df=params.predict_data,
                                                     index_columns_p=params.index_columns_predict,
                                                     drop_columns_p=params.drop_columns_predict,
                                                     target_column=params.target_column,
                                                     keep_columns_p=params.keep_columns_predict,
                                                     outlier_threshold_p=params.outlier_threshold_predict)

                print('Predict data:')
                print(predict_data)
                print(type(predict_data))

                dat = align_dataframes(df1=processed_data, df2=predict_data)
                print(type(dat))

                forecasted_data = predict(params.single_year_prediction,
                                          trained_data=processed_data,
                                          predict_data=dat,
                                          trained_model=model,
                                          target_column=params.target_column,
                                          output_folder=params.output_folder,
                                          FinalModelParameters=None,
                                          index_col=params.index_columns,
                                          skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                          basic_model=params.basic_model)

                print(forecasted_data)

                eval_model(data_used_to_predict=dat,
                           data_contains_truth_values_only=params.validation_data,
                           model_predicted_data=forecasted_data,
                           model=model,
                           target_column=params.target_column,
                           output_folder=params.output_folder)


                return


            data = filter_data(original_data=processed_data,
                               best_features_model=selected_features_model,
                               best_features_f_regressor=selected_f_regressor,
                               best_features_rfe=selected_rfe,
                               output_folder=params.output_folder,
                               target_column=params.target_column,
                               best_features_sfs=selected_sfs,
                               best_features_mutual_info=selected_mutual_info,
                               model=model)

            ## FINAL DATA ANALYSIS ##
            df_final_to_model, transformations = pre_forecast_data_analysis(data=data,
                                                                            regression_method=model,
                                                                            target_column=params.target_column,
                                                                            threshold=params.threshold,
                                                                            threshold_corr=params.threshold_corr,
                                                                            output_folder=params.output_folder,
                                                                            index_col=params.index_columns)

            print("Final data to be modelled:")
            print(df_final_to_model)
            print('Conducting hyperparamter optimisation')

            ## HYPERPARAMETER OPTIMISATION ##
            hyperparameters = select_param(data=df_final_to_model,
                                           target_column=params.target_column,
                                           model=model)

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
                                                 outlier_threshold_p=params.outlier_threshold_predict)

            print('Predict data:')
            print(predict_data)
            print(type(predict_data))

            dat = align_dataframes(df1=df_final_to_model, df2=predict_data)
            print(type(dat))

            final_predict_data = apply_transformations(predict_data=dat,
                                                       transformations=transformations)

            print(final_predict_data)
            print(model)

            forecasted_data = predict(params.single_year_prediction,
                                      trained_data=df_final_to_model,
                                      predict_data=final_predict_data,
                                      trained_model=model,
                                      target_column=params.target_column,
                                      output_folder=params.output_folder,
                                      FinalModelParameters=hyperparameters,
                                      index_col=params.index_columns,
                                      skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                      basic_model=params.basic_model)

            print(forecasted_data)

            eval_model(data_used_to_predict=final_predict_data,
                       data_contains_truth_values_only=params.validation_data,
                       model_predicted_data=forecasted_data,
                       model=model,
                       target_column=params.target_column,
                       output_folder=params.output_folder)


        elif params.multiple_year_prediction is not None:
            if params.categorical_data is not None:
                ## PROCESS DATA ##
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

                print('Checking if data is numeric')
                all_numeric = processed_data.applymap(np.isreal).all().all()
                print(all_numeric)

                preprocessed_df, transformations_ = process_data_pipeline(df=processed_data,
                                                        numerical_features=params.numerical_features,
                                                        categorical_features=params.categorical_features,
                                                        target_column=params.target_column,
                                                        output_folder=params.output_folder)

                ## FEATURE SELECTION ##
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

                ## FINAL DATA ANALYSIS ##
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


                forecasted_data = predict(params.single_year_prediction,
                                          trained_data=df_final_to_model,
                                          predict_data=final_predict_data,
                                          trained_model=model,
                                          target_column=params.target_column,
                                          output_folder=params.output_folder,
                                          FinalModelParameters=hyperparameters,
                                          index_col=params.index_columns,
                                          skip_hyperparameter_optimisation=params.skip_hyperparameter_optimisation,
                                          basic_model=params.basic_model,
                                          multiple_year_prediction=params.multiple_year_prediction,
                                          categorical_data=params.categorical_data)

                print(forecasted_data)

                eval_model(data_used_to_predict=final_predict_data,
                           data_contains_truth_values_only=params.validation_data,
                           model_predicted_data=forecasted_data,
                           model=model,
                           target_column=params.target_column,
                           output_folder=params.output_folder,
                           categorical_target=params.categorical_target)

                return

#todo indexing still causing massive issues, speak to someone about it
    warnings.filterwarnings("default")

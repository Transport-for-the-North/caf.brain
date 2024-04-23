# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from caf.ml.inputs.cafml_inputs import CarAccessInputs
from caf.ml.car_access_land_use.process_data_class import DataProcessor
from caf.ml.car_access_land_use.feature_selection import feature_selection_cv, filter_data
from caf.ml.functions.hyper_optim_gridsearch import select_param
from caf.ml.functions.forecast_model_functions import predict, process_data_loaded_model, process_forecast_data, align_dataframes
from caf.ml.functions.save_model import save_model_and_parameters, load_model_and_parameters
from caf.ml.functions.pre_forecast_data_analysis import pre_forecast_data_analysis, apply_transformations
# todo add interpolation function from lvu to dataprocessor class


def main(params: CarAccessInputs):
    ######## MODEL AND DATA SAVED ########
    if params.saved_model is not None:
        ## LOAD MODEL AND DATA ##
        regression_method, hyperparameters, trained_data, transformations = load_model_and_parameters(
            params.output_folder)

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

        dat = align_dataframes(df1=trained_data, df2=predict_data)
        print(type(dat))

        final_predict_data = apply_transformations(predict_data=dat,
                                                   transformations=transformations)

        print(final_predict_data)

        ## FINAL FORECAST ##
        forecasted_data = predict(params.single_year_prediction,
                                  trained_data=trained_data,
                                  predict_data=final_predict_data,
                                  trained_model=regression_method,
                                  target_column=params.target_column,
                                  output_folder=params.output_folder,
                                  FinalModelParameters=hyperparameters,
                                  index_col=params.index_columns)
        print(forecasted_data)

    ######## SIMPLE FEATURE SELECTION ########
    if params.simple_feature_selection is not None:

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
        selected_features_model, selected_f_classif, selected_rfe, model = feature_selection_cv(
            data=processed_data,
            model_type=params.model_type,
            cv_method=params.cv_method,
            splits=params.splits,
            repeats=params.repeats,
            target_column=params.target_column)
        print(model)

        data = filter_data(original_data=processed_data,
                           best_features_model=selected_features_model,
                           best_features_f_classif=selected_f_classif,
                           best_features_rfe=selected_rfe,
                           output_folder=params.output_folder,
                           target_column=params.target_column)

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
        save_model_and_parameters(model,
                                  hyperparameters,
                                  transformations,
                                  output_folder=params.output_folder)

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
                                  index_col=params.index_columns)

        print(forecasted_data)


    ######## EXHAUSTIVE FEATURE SELECTION ########
    elif params.simple_feature_selection is None:

        if params.saved_model is None:
            # todo explore more extensive feature selection - main issue is run time

            warnings.filterwarnings("default")

import sys
import numpy as np
import pandas as pd

sys.path.extend(r"C:\Users\Liberty\Documents\GitHub\caf.ml\src")
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from caf.ml.inputs.cafml_inputs import CarAccessInputs
from caf.ml.car_access_land_use.process_data_class import DataProcessor
from caf.ml.car_access_land_use.feature_selection import feature_selection_cv, filter_data
from caf.ml.functions.hyper_optim_gridsearch import select_param
from caf.ml.functions.forecast_model_functions import predict, process_data_loaded_model, \
    process_forecast_data, match_columns
from caf.ml.functions.save_model import save_model_and_parameters, load_model_and_parameters
from caf.ml.functions.pre_forecast_data_analysis import pre_forecast_data_analysis


# todo add interpolation function from lvu to dataprocessor class
# todo refer to flow in note book for final forecasting part


def main(params: CarAccessInputs):
    if params.simple_feature_selection is not None:

        ######## MODEL AND DATA SAVED ########
        if params.saved_model is not None:

            #### LOAD MODEL AND DATA ####
            regression_method, hyperparameters, trained_data = load_model_and_parameters(
                params.output_folder)

            print(regression_method)
            print(type(regression_method))

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

            print(trained_data)
            print(trained_data.shape)
            print(trained_data.dtypes)
            print(predict_data)
            print(predict_data.shape)
            print(predict_data.dtypes)

            predict_data = match_columns(trained_data=trained_data, predict_data=predict_data)

            #### FINAL FORECAST/PREDICT ####
            forecasted_data = predict(params.single_year_prediction,
                                      trained_data=trained_data,
                                      predict_data=predict_data,
                                      trained_model=regression_method,
                                      target_column=params.target_column,
                                      vif_threshold=params.threshold,
                                      threshold_corr=params.threshold_corr,
                                      output_folder=params.output_folder,
                                      FinalModelParameters=hyperparameters)

            print(forecasted_data)

        ######## MODEL AND DATA NOT SAVED ########
        elif params.saved_model is None:

            #### PROCESS DATA #####
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
                                           params.outlier_threshold
                                           ).data

            print('Checking if data is numeric')
            all_numeric = processed_data.applymap(np.isreal).all().all()
            print(all_numeric)

            #### FEATURE SELECTION #####
            selected_features_model, selected_f_classif, selected_rfe, model = feature_selection_cv(
                data=processed_data,
                model_type=params.model_type,
                cv_method=params.cv_method,
                splits=params.splits,
                repeats=params.repeats,
                target_column=params.target_column)

            data = filter_data(original_data=processed_data,
                               best_features_model=selected_features_model,
                               best_features_f_classif=selected_f_classif,
                               best_features_rfe=selected_rfe,
                               output_folder=params.output_folder,
                               target_column=params.target_column)

            #### FINAL DATA ANALYSIS ####
            df_final_to_model = pre_forecast_data_analysis(data=data,
                                                           regression_method=model,
                                                           target_column=params.target_column,
                                                           threshold=params.threshold,
                                                           threshold_corr=params.threshold_corr,
                                                           output_folder=params.output_folder)

            print("Final data to be modelled:")
            print(df_final_to_model)
            print('Conducting hyperparamter optimisation')

            #### HYPERPARAMETER OPTIMISATION #####
            hyperparameters = select_param(data=df_final_to_model,
                                           target_column=params.target_column,
                                           regression_method=model,
                                           model_type=None)

            #### SAVE MODEL #####
            save_model_and_parameters(model,
                                      hyperparameters,
                                      output_folder=params.output_folder)

            #### FINAL PREDICTION #####
            forecasted_data = predict(params.single_year_prediction,
                                      data_p=params.predict_data,
                                      index_columns_p=params.index_columns_predict,
                                      drop_columns_p=params.drop_columns_predict,
                                      target_column=params.target_column,
                                      keep_columns_p=params.keep_columns_predict,
                                      outlier_threshold_p=params.outlier_threshold_predict,
                                      FinalDataframe=df_final_to_model,
                                      FinalModel=regression_method,
                                      FinalModelParameters=hyperparameters,
                                      output_folder=params.output_folder,
                                      year_range=params.year_range,
                                      vif_threshold=params.threshold,
                                      threshold_corr=params.threshold_corr)

            print(forecasted_data)

    warnings.filterwarnings("default")


if __name__ == "__main__":
    '''
    --------------------------------------------------------------------------                     
    Process Data:
    :param file_path: path to data if data is an individual file
    :param folder_path: path to data if data is a folder of multiple files
    :param index_columns: columns (titles) within your data that become the index.
                          e.g year column, an essential column but doesnt 
                          need to be modeled
    :param drop_columns: columns (titles) to remove from your input data
    :param target_column: the column (title) we want to forecast
    :param outlier_threshold: threshold to check data against in order to 
                              remove outliers.
    :param wide_format: Either yes or none. Yes if your data is wide and needs                    
                        to be made long
    if wide_format is yes:
                          :param variable_name: (var_name) name of column
                          that will be contain column names from original 
                          dataframe
                          :param value_name: name of column that will contain 
                          the values associated with melted dataframe
                          see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html#pandas.melt

    --------------------------------------------------------------------------                     
    Algorithim Evaluation:
    :param models_to_test: A list of models to evaluate. The best model for the
                           data will be selected. Available models:
                           RANDOM_FOREST
                           EXTRA_TREES
                           GRADIENT_BOOSTING
                           ADABOOST
                           BAGGING
                           SVR
                           KNN
                           RIDGE
                           LASSO
                           ELASTICNET
                           LINEAR_REGRESSION
                           DECISION_TREE
                           NEURAL_NETWORK
    --------------------------------------------------------------------------                             
    Feature Selection:
    :param grid_search: Either none or yes. If yes, grid search feature selection
                        conducted. If none then alterantive feature selection
                        (selectfrommodel, pvalue, recursive)

    If grid_search none: 
        :param target_column: the column (title) we want to forecast
        :param model_type: Algorthim used within the cross validation element
                           of feature selection and future modelling. 
                           Available models:
                           RANDOM_FOREST
                           EXTRA_TREES
                           GRADIENT_BOOSTING
                           ADABOOST
                           BAGGING
                           SVR
                           KNN
                           RIDGE
                           LASSO
                           ELASTICNET
                           LINEAR_REGRESSION
                           DECISION_TREE
                           NEURAL_NETWORK
        :param cv_method: Cross validation method used when conducting feature
                          selection and further modelling. Available models:
                          kfold
                          stratifiedkfold
                          repeatedkfold
                          repeatedstratifiedkfold
        :param splits: Number of splits to do implemented into cross validation.
        :param repeated: For repeatedkfold and repeatedstratidiedkfold. 
                         Number of repeates for specified algorithms


    If grid_search yes: 

    '''
    params = CarAccessInputs(
        #### PROCESS DATA ####
        # imports
        x_path=Path(r"E:\caf.ml\Prediction_Model\test_data\census_2011.csv"),
        y_path=None,
        folder_path=None,

        # data sorting imports
        index_columns=['geo_code'],
        drop_columns=['sum of all cars or vans in the area'],
        keep_columns=None,
        target_column='No cars or vans in household',
        output_folder=Path(r"E:\caf.ml\Prediction_Model\test_data\output"),

        # wide to long imports
        wide_format=None,
        variable_name=None,
        value_name=None,

        # optional imports
        outlier_threshold=None,

        #### FEATURE SELECTION ####
        # feature selection methods
        simple_feature_selection='yes',
        feature_selection_exhaustive=None,

        # feature selection processing
        process_data_used=True,
        model_type=None,
        cv_method=None,
        splits=None,
        repeats=None,

        #### PREDICT DATA ####
        single_year_prediction='yes',
        predict_data=Path(r"E:\caf.ml\Prediction_Model\test_data\census_2021.csv"),
        year_range=None,

        # data sorting imports for predict data
        index_columns_predict=['geo_code'],
        drop_columns_predict=['sum of all cars or vans in the area'],
        keep_columns_predict=None,
        outlier_threshold_predict=None,

        #### STORE MODEL ####
        # MODEL_FILE_PATH = r"E:\caf.ml\Prediction_Model\test_data\saved_model\model.pkl"
        # PARAMETERS_FILE_PATH = r"E:\caf.ml\Prediction_Model\test_data\saved_model\parameters.pkl"
        saved_model=None,

        threshold=10,
        threshold_corr=0.7

    )
    main(params)

import sys
sys.path.extend(r"C:\Users\Liberty\Documents\GitHub\caf.ml\src")
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from caf.ml.inputs.cafml_inputs import CarAccessInputs
from caf.ml.car_access_land_use.process_data_class import DataProcessor
from caf.ml.car_access_land_use.feature_selection import feature_selection_, apply_feature_selection
from caf.ml.functions.feature_selection_hyper_optim import feature_selection_with_optimization
from caf.ml.functions.hyper_optim_gridsearch import select_param


def main(params: CarAccessInputs):
    ######## PROCESS DATA #########
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
                                   ).data


    #todo add interpolation function from lvu to dataprocessor class
    ######## SIMPLE FEATURE SELECTION #########
    if params.simple_feature_selection is not None:
        # (30 min run time) non grid search feature selection
        selected_feature_indices = feature_selection_(processed_data,
                                                      params.target_column,
                                                      params.model_type,
                                                      params.cv_method,
                                                      params.splits,
                                                      params.repeats)
        df_final = apply_feature_selection(selected_feature_indices)
        print("Best features for model determined to be: %s", df_final)
        model = select_param(df_final,
                             params.target_column,
                             params.model_type)

    ######## FEATURE SELECTION WITH HYPERPARAMETER OPTIMISATION ########
    elif params.simple_feature_selection is None:
        model, df_final = feature_selection_with_optimization(processed_data,
                                                              params.splits,
                                                              params.repeats,
                                                              params.model_type,
                                                              params.target_column,
                                                              params.cv_method)
        print("Best features for model determined to be: %s", df_final)
        print("Best model determined to be: %s", model)
    warnings.filterwarnings("default")

    ######## FINAL MODEL FORECAST ########




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
                        x_path=Path(r"E:\caf.ml\data_process_function\test_data\cb_tfn_v12.csv"),
                        y_path=None,
                        folder_path=None,

                        # data sorting imports
                        index_columns=None,
                        drop_columns=None,
                        keep_columns=None,
                        target_column=None,
                        output_folder=None,

                        # wide to long imports
                        wide_format=None,
                        variable_name=None,
                        value_name=None,

                        # optional imports
                        outlier_threshold=None,


                        #### FEATURE SELECTION ####
                        simple_feature_selection=None,
                        model_type=None,
                        cv_method=None,
                        splits=None,
                        repeats=None,





                        hp_optimisation=None,
                        # output file path
                        folder=Path(r"E:\caf.ml\data_process_function\car_model_results"),
    )
    main(params)

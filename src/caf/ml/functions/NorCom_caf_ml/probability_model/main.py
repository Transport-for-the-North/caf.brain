# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 10/10/2024
Original author: Adil Zaheer
"""
import os
import pandas as pd
from caf.ml.functions.NorCom_caf_ml.probability_model.final_model_functions import refined_cafml_model
from caf.ml.functions.NorCom_caf_ml.probability_model.functions import (encode_and_sort,
                                                                        generate_stats_model,
                                                                        refined_data_processor_function,
                                                                        generate_sklearn_model,
                                                                        generate_svm,
                                                                        generate_cafml_model)
from caf.ml.functions.NorCom_caf_ml.probability_model.inputs import NorCom_probability_model_inputs_cafml

def main(params: NorCom_probability_model_inputs_cafml):

    tidy_processed_path = os.path.join(params.output_folder, 'initial_processed_data.csv')
    if os.path.exists(tidy_processed_path):
        print('Tidy_processed_data exists so is being read in')
        processed_data = pd.read_csv(tidy_processed_path, low_memory=False)
        processed_data.set_index(params.index_columns, inplace=True)
        print("Loaded processed data:\n", processed_data.head())

        index_columns_csv_path = os.path.join(params.output_folder, 'index_columns_csv.csv')
        index_columns_df = pd.read_csv(index_columns_csv_path)

    else:
        processed_data, index_columns_df = refined_data_processor_function(df=params.classified_build,
                                                                           target_column=params.target_column,
                                                                           index_columns=params.index_columns,
                                                                           categorical_features=params.categorical_features,
                                                                           numerical_features=params.numerical_features,
                                                                           output_folder=params.output_folder,
                                                                           column_name_to_drop_rows=params.column_name_to_drop_rows,
                                                                           value_in_row=params.value_in_row,
                                                                           weight_column=params.weight_column)

    training_dat_path = os.path.join(params.output_folder, 'training_data.csv')
    if os.path.exists(training_dat_path):
        print('Final training data exists so is being read in')
        training_df = pd.read_csv(training_dat_path)
        training_df.set_index(params.index_columns, inplace=True)

        test_df = pd.read_csv(os.path.join(params.output_folder, 'test_data.csv'))
        test_df.set_index(params.index_columns, inplace=True)

        validation_df = pd.read_csv(os.path.join(params.output_folder, 'validation_data.csv'))
        validation_df.set_index(params.index_columns, inplace=True)

        print("Loaded training data:\n", training_df.head())
        print("Loaded test data:\n", test_df.head())
        print("Loaded validation data:\n", validation_df.head())

    else:
        training_df, test_df, validation_df = encode_and_sort(df=processed_data,
                                                              target_column=params.target_column,
                                                              output_folder=params.output_folder,
                                                              categorical_feat=params.categorical_features,
                                                              training_year=params.training_year,
                                                              weight_column=params.weight_column,
                                                              binary_prediction=params.binary_prediction)

    if params.stats_model is not None:
        generate_stats_model(training_df=training_df,
                             test_df=test_df,
                             validation_df=validation_df,
                             target_column=params.target_column,
                             output_folder=params.output_folder)

    elif params.sklearn_model is not None:
        generate_sklearn_model(training_df=training_df,
                               test_df=test_df,
                               validation_df=validation_df,
                               target_column=params.target_column,
                               output_folder=params.output_folder)

    elif params.svm_adaptation is not None:
        generate_svm(training_df=training_df,
                     test_df=test_df,
                     validation_df=validation_df,
                     target_column=params.target_column,
                     output_folder=params.output_folder)

    elif params.cafml_version is not None:
        generate_cafml_model(training_df=training_df,
                             test_df=test_df,
                             validation_df=validation_df,
                             index_columns_df=index_columns_df,
                             target_column=params.target_column,
                             output_folder=params.output_folder,
                             weight_column=params.weight_column,
                             improve_data=params.improve_data,
                             model_to_use=params.model_choice,
                             index_columns=params.index_columns,
                             binary_prediction=params.binary_prediction,
                             skip_feature_selection=params.skip_feature_selection)

    elif params.refined_cafml is not None:
        refined_cafml_model(training_df=training_df,
                            test_df=test_df,
                            validation_df=validation_df,
                            target_column=params.target_column,
                            output_folder=params.output_folder,
                            weight_column=params.weight_column,
                            model_to_use=params.model_choice,
                            index_columns=params.index_columns,
                            binary_prediction=params.binary_prediction)


    return

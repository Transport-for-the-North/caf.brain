# -*- coding: utf-8 -*-
"""
Created on: 10/8/2024
Original author: Adil Zaheer
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from caf.ml.backlog.functions_to_be_processed import process_data_pipeline
from caf.ml.backlog.functions_to_be_processed import process_forecast_data
from caf.ml.backlog.functions_to_be_processed import apply_transformations
from caf.ml.backlog.functions_to_be_processed import DataProcessor
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.MODELS.requires_update_tf_nn_model.tf_functions import train_model, calculate_coefficient_stats, train_improved_model
from caf.ml.backlog.old_inputs.cafml_inputs import TensorFlowModelInputs


def main(params: TensorFlowModelInputs):
    processed_data = DataProcessor(x=params.training_data,
                                   y=None,
                                   folder_path=None,
                                   index_columns=params.index_columns,
                                   drop_columns=params.drop_columns,
                                   keep_columns=None,
                                   target_column=params.target_column,
                                   output_folder=params.output_folder,
                                   wide_format=None,
                                   variable_name=None,
                                   value_name=None,
                                   outlier_threshold=None,
                                   categorical_target=params.categorical_target,
                                   column_name_to_drop_rows=params.column_name_to_drop_rows,
                                   value_in_row=params.value_in_row).data

    processed_training, transformations_ = process_data_pipeline(df=processed_data,
                                                                 numerical_features=params.numerical_features,
                                                                 categorical_features=params.categorical_features,
                                                                 target_column=params.target_column,
                                                                 output_folder=params.output_folder)

    predict_data = process_forecast_data(df=params.prediction_data,
                                         index_columns_p=params.index_columns,
                                         drop_columns_p=params.drop_columns,
                                         target_column=params.target_column,
                                         keep_columns_p=None,
                                         outlier_threshold_p=None,
                                         categorical_target=params.categorical_target,
                                         output_folder=params.output_folder)

    processed_predict = apply_transformations(predict_data=predict_data,
                                              transformations=transformations_,
                                              target_column=params.target_column,
                                              numerical_features=params.numerical_features,
                                              categorical_features=params.categorical_features,
                                              output_folder=params.output_folder,
                                              training_data=processed_training,
                                              features_to_transform=None)

    validate = pd.read_csv(params.validation_data)
    validate = validate.set_index(params.index_columns)

    x = processed_training.drop(columns=params.target_column, axis=1)
    y = processed_training[params.target_column]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False,
                                                        random_state=42)

    y_train = pd.to_numeric(y_train, errors='coerce')


    if params.simple_model is not None:
        print('Logit model running')
        model, predictions = train_model(X_train=X_train,
                                         y_train=y_train,
                                         X_predict=processed_predict,
                                         y_validate=validate,
                                         output_folder=params.output_folder,
                                         index_col=params.index_columns,
                                         target_column=params.target_column,
                                         predict_data=predict_data)
    else:
        print('NN model running')
        model, predictions = train_improved_model(X_train=X_train,
                                                  y_train=y_train,
                                                  X_predict=processed_predict,
                                                  y_validate=validate,
                                                  output_folder=params.output_folder,
                                                  index_col=params.index_columns,
                                                  target_column=params.target_column,
                                                  predict_data=predict_data)


    coef_stats = calculate_coefficient_stats(model, X_train, y_train, output_folder=params.output_folder)
    print("\nCoefficient Statistics:")
    print(coef_stats.to_string(index=False))

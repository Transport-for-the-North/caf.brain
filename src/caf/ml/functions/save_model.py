# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
import pickle

import joblib
import pandas as pd


def save_model_and_parameters(regression_method,
                              hyperparameters,
                              transformations,
                              output_folder):

    regression_method_file_path = output_folder / 'regression_method.pkl'
    hyperparameters_file_path = output_folder / 'hyperparameters.pkl'
    transformations_file_path = output_folder / 'transformations.pkl'

    joblib.dump(regression_method, regression_method_file_path)
    joblib.dump(hyperparameters, hyperparameters_file_path)
    joblib.dump(transformations, transformations_file_path)

    print(f"Model saved to: {regression_method_file_path}")
    print(f"Parameters saved to: {hyperparameters_file_path}")
    print(f'Transformations saved to: {transformations_file_path}')


def load_model_and_parameters(output_folder):
    regression_method_file_path = output_folder / 'regression_method.pkl'
    hyperparameters_file_path = output_folder / 'hyperparameters.pkl'
    dataframe_file_path = output_folder / 'final_data_ready_to_model.csv'
    transformations_file_path = output_folder / 'transformations.pkl'

    regression_method = joblib.load(regression_method_file_path)
    parameters = joblib.load(hyperparameters_file_path)
    df_final = pd.read_csv(dataframe_file_path)
    transformations = joblib.load(transformations_file_path)

    print(f"Model loaded from: {regression_method_file_path}")
    print(f"Parameters loaded from: {hyperparameters_file_path}")
    print(f"DataFrame loaded from: {dataframe_file_path}")
    print(f'Transformations loaded from: {transformations_file_path}')

    return regression_method, parameters, df_final, transformations

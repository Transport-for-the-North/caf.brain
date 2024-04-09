# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import joblib
import pandas as pd


def save_model_and_parameters(regression_method,
                              hyperparameters,
                              output_folder):

    regression_method_file_path = output_folder / 'regression_method.pkl'
    hyperparameters_file_path = output_folder / 'hyperparameters.pkl'

    joblib.dump(regression_method, regression_method_file_path)
    joblib.dump(hyperparameters, hyperparameters_file_path)

    print(f"Model saved to: {regression_method_file_path}")
    print(f"Parameters saved to: {hyperparameters_file_path}")


def load_model_and_parameters(output_folder):
    regression_method_file_path = output_folder / 'regression_method.pkl'
    hyperparameters_file_path = output_folder / 'hyperparameters.pkl'
    dataframe_file_path = output_folder / 'final_data_to_model.csv'

    regression_method = joblib.load(regression_method_file_path)
    parameters = joblib.load(hyperparameters_file_path)
    df_final = pd.read_csv(dataframe_file_path)

    print(f"Model loaded from: {regression_method_file_path}")
    print(f"Parameters loaded from: {hyperparameters_file_path}")
    print(f"DataFrame loaded from: {dataframe_file_path}")

    return regression_method, parameters, df_final

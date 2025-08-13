# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
# Built-Ins
import os
import pickle

# Third Party
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


def save_model_and_parameters(
    regression_method,
    hyperparameters,
    transformations,
    output_folder,
    skip_data_analysis,
    skip_hyperparameter_optimisation,
):

    if skip_data_analysis is not None:
        regression_method_file_path = output_folder / "regression_method.pkl"
        joblib.dump(regression_method, regression_method_file_path)
        print(f"Model saved to: {regression_method_file_path}")
        hyperparameters_file_path = output_folder / "hyperparameters.pkl"
        joblib.dump(hyperparameters, hyperparameters_file_path)
        print(f"Parameters saved to: {hyperparameters_file_path}")

    if skip_hyperparameter_optimisation is not None:
        regression_method_file_path = output_folder / "regression_method.pkl"
        transformations_file_path = output_folder / "transformations.pkl"
        joblib.dump(regression_method, regression_method_file_path)
        joblib.dump(transformations, transformations_file_path)
        print(f"Model saved to: {regression_method_file_path}")
        print(f"Transformations saved to: {transformations_file_path}")

    if regression_method == LinearRegression():
        regression_method_file_path = output_folder / "regression_method.pkl"
        transformations_file_path = output_folder / "transformations.pkl"
        joblib.dump(regression_method, regression_method_file_path)
        joblib.dump(transformations, transformations_file_path)
        print(f"Model saved to: {regression_method_file_path}")
        print(f"Transformations saved to: {transformations_file_path}")

    else:
        regression_method_file_path = output_folder / "regression_method.pkl"
        hyperparameters_file_path = output_folder / "hyperparameters.pkl"
        transformations_file_path = output_folder / "transformations.pkl"

        joblib.dump(regression_method, regression_method_file_path)
        joblib.dump(hyperparameters, hyperparameters_file_path)
        joblib.dump(transformations, transformations_file_path)

        print(f"Model saved to: {regression_method_file_path}")
        print(f"Parameters saved to: {hyperparameters_file_path}")
        print(f"Transformations saved to: {transformations_file_path}")


def load_model_and_parameters(output_folder):
    regression_method_file_path = output_folder / "regression_method.pkl"
    hyperparameters_file_path = output_folder / "hyperparameters.pkl"
    dataframe_file_path = output_folder / "final_data_ready_to_model.csv"
    transformations_file_path = output_folder / "transformations.pkl"
    data_analysis_data_file_path = output_folder / "data_analysis_dataframe.csv"

    regression_method = joblib.load(regression_method_file_path)
    df_final = pd.read_csv(dataframe_file_path)
    data_analysis_data = pd.read_csv(data_analysis_data_file_path)
    print(f"Model loaded from: {regression_method_file_path}")
    print(f"Parameters loaded from: {hyperparameters_file_path}")
    print(f"DataFrame loaded from: {dataframe_file_path}")

    if hyperparameters_file_path.exists():
        parameters = joblib.load(hyperparameters_file_path)
        print(f"Parameters loaded from: {hyperparameters_file_path}")
    else:
        parameters = None
        print(f"Parameters file not found at: {hyperparameters_file_path}")

    if transformations_file_path.exists():
        transformations = joblib.load(transformations_file_path)
        print(f"Transformations loaded from: {transformations_file_path}")
    else:
        transformations = None
        print(f"Transformations file not found at: {transformations_file_path}")

    return regression_method, parameters, df_final, transformations, data_analysis_data

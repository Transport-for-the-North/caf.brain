# -*- coding: utf-8 -*-
"""
Created on: 2/16/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import os
from caf.ml.functions.process_data_functions import (index_sorter,
                                                     function_remove_spaces,
                                                     convert_to_dataframe,
                                                     find_numeric_target_column,
                                                     process_data_numeric,
                                                     handle_nans_and_duplicates,
                                                     remove_and_export_outliers)


def process_data_loaded_model(df,
                              index_columns,
                              drop_columns,
                              target_column,
                              keep_columns,
                              outlier_threshold):


    # Convert column names to strings and data types to float
    df.columns = df.columns.astype(str)

    dat = index_sorter(df, index_columns=index_columns, drop_columns=drop_columns)
    dat = dat.astype(float)
    dat = function_remove_spaces(dat)
    dat = convert_to_dataframe(dat)
    dat = find_numeric_target_column(dat, target_column=target_column)
    dat = process_data_numeric(dat, keep_columns=keep_columns)
    dat = handle_nans_and_duplicates(dat)
    dat = remove_and_export_outliers(dat, outlier_threshold=outlier_threshold)
    data = convert_to_dataframe(dat)

    return data


def process_forecast_data(df,
                          index_columns_p,
                          drop_columns_p,
                          target_column,
                          keep_columns_p,
                          outlier_threshold_p):

    x_ = pd.read_csv(df, low_memory=False)
    # Convert column names to strings and data types to float
    x_.columns = x_.columns.astype(str)
    dat = index_sorter(x_, index_columns=index_columns_p, drop_columns=drop_columns_p)
    dat = dat.astype(float)
    dat = function_remove_spaces(dat)
    dat = convert_to_dataframe(dat)
    dat = find_numeric_target_column(dat, target_column=target_column)
    dat = process_data_numeric(dat, keep_columns=keep_columns_p)
    dat = handle_nans_and_duplicates(dat)
    dat = remove_and_export_outliers(dat, outlier_threshold=outlier_threshold_p)
    data = convert_to_dataframe(dat)

    return data


def match_columns(trained_data, predict_data):
    # Ensure predict_data contains all columns present in trained_data
    missing_columns = set(trained_data.columns) - set(predict_data.columns)
    if missing_columns:
        for column in missing_columns:
            predict_data[column] = None

    # Check data types and index alignment
    for column in trained_data.columns:
        if trained_data[column].dtype != predict_data[column].dtype:
            # Convert data type in predict_data to match trained_data
            predict_data[column] = predict_data[column].astype(trained_data[column].dtype)

    # Align indices
    predict_data = predict_data.reindex(trained_data.index)

    return predict_data



def apply_feature_selection_single_year(trained_data, predict_data):
    if isinstance(trained_data[0], (list, tuple)):
        # Convert the list of selected features to a flat list
        selected_features = [item for sublist in trained_data for item in sublist]
    else:
        # Get the column names from the feature selection DataFrame
        selected_features = trained_data.columns

    # Filter the columns of the DataFrame to be applied based on the selected features
    df_final = predict_data[selected_features]

    return df_final


def predict(single_year_prediction,
            trained_data,
            predict_data,
            trained_model,
            target_column,
            vif_threshold,
            threshold_corr,
            output_folder,
            FinalModelParameters):

    predictions = None
    if single_year_prediction is not None:

        # Check if the prediction data matches the training data
        if not predict_data.columns.equals(trained_data.columns) or \
                not predict_data.dtypes.equals(trained_data.dtypes) or \
                not predict_data.index.names == trained_data.index.names or \
                not all(predict_data.index.levels[i].equals(trained_data.index.levels[i]) for i in
                        range(len(predict_data.index.levels))):
            raise ValueError("Prediction data does not match the training data. Check columns, dtypes, geography, or index.")

        # Get the actual model class from the enum
        trained_model_class = trained_model.value

        trained_model = trained_model_class(**FinalModelParameters)
        trained_model.fit(trained_data.drop(target_column, axis=1), trained_data[target_column])
        predictions = trained_model.predict(predict_data)

        # Create a DataFrame to store the predictions
        prediction_df = pd.DataFrame({target_column: predictions}, index=predict_data.index)

        # Save the predictions to a CSV file
        prediction_file_path = os.path.join(output_folder, 'predictions.csv')
        prediction_df.to_csv(prediction_file_path)
        print(f"Predictions saved to: {prediction_file_path}")

    else:
        pass

    return predictions


'''
    # Generate forecasts
    if year_range:
        start_year, end_year = year_range
        forecast_years = range(start_year, end_year + 1)
    else:
        # If year_range is not provided, use the range of years in the input (forecast) data
        forecast_years = range(data.index.min(), data.index.max() + 1)

    forecasts = []
    for year in forecast_years:

        # Make predictions using FinalModel with optimized hyperparameters
        predictions = FinalModel(**FinalModelParameters).predict(prediction_data)

        # Store predictions along with index (year)
        forecasts.extend(zip([year] * len(predictions), predictions))

    # Convert forecasts to DataFrame
    forecast_df = pd.DataFrame(forecasts, columns=['Year', 'Predicted'])

    # Save forecasted results
    forecast_file_path = os.path.join(output_folder, 'forecasts.csv')
    forecast_df.to_csv(forecast_file_path, index=False)
    print(f"Forecasted results saved to: {forecast_file_path}")

    # Evaluate forecast accuracy
    if FinalDataframe is not None:
        evaluate_forecast_accuracy(forecast_df, FinalDataframe[target_column])

    return forecast_df
'''


def evaluate_forecast_accuracy(forecast_df: pd.DataFrame, actual_values: pd.Series):
    # Calculate metrics
    mse = mean_squared_error(actual_values, forecast_df['Predicted'])
    r2 = r2_score(actual_values, forecast_df['Predicted'])

    # Print evaluation metrics
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")

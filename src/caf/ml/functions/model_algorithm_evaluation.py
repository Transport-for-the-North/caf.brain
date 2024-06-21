# -*- coding: utf-8 -*-
"""
Created on: 2/13/2024
Original author: Adil Zaheer
"""
import os

import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import pandas as pd
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from caf.ml.inputs.cafml_inputs import Models
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, \
    mean_absolute_percentage_error, mean_squared_log_error, explained_variance_score, \
    median_absolute_error, accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def select_model(x: pd.DataFrame, y: pd.DataFrame, models_to_test: list[Models], output_folder):

    acc = {}
    score = 0
    return_model = None
    for model_enum in models_to_test:
        model_instance = model_enum.value()
        model_name = model_enum.name.lower()

        scores_r2 = cross_val_score(model_instance, x, y, cv=5, scoring="r2")
        scores_mse = -cross_val_score(model_instance, x, y, cv=5, scoring="neg_mean_squared_error")

        mean_r2 = scores_r2.mean()
        mean_mse = scores_mse.mean()

        acc[model_name] = {'R-squared': mean_r2, 'MSE': mean_mse}
        if mean_r2 > score:
            score = mean_r2
            return_model = model_instance
    print(f"Best model score: {score}")
    evaluation_df = pd.DataFrame.from_dict(acc, orient='index')

    output_filename = 'pre_transformation_model_evaluation_results.csv'
    output_path = os.path.join(output_folder, output_filename)
    evaluation_df.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"Pre-transformation model evaluation results exported to: {output_path}")

    return return_model


def eval_model(data_used_to_predict,
               data_contains_truth_values_only,
               model_predicted_data,
               model,
               target_column,
               output_folder,
               categorical_target):
    y_truth = None

    if data_contains_truth_values_only is not None:
        dat = pd.read_csv(data_contains_truth_values_only)
        if target_column in dat.columns:
            y_truth = dat[target_column]
    elif target_column in data_used_to_predict.columns:
        y_truth = data_used_to_predict[target_column]
    else:
        print(f"Validation data or target column '{target_column}' not available.")
        return

    model_predicted_data = pd.to_numeric(model_predicted_data, errors='coerce')

    if categorical_target is not None:
        # Calculate accuracy, precision, recall, F1-score for each class
        accuracy = accuracy_score(y_truth, model_predicted_data)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_truth,
                                                                       model_predicted_data,
                                                                       average='weighted')
        conf_matrix = confusion_matrix(y_truth, model_predicted_data)
        class_report = classification_report(y_truth, model_predicted_data)

        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': fscore
        }

        metrics_df = pd.DataFrame([metrics_dict])
        metrics_df.to_csv(os.path.join(output_folder, 'model_evaluation_metrics.csv'), index=False)

        return

    metrics_dict = {
        'Model': model,
        'RMSE': mean_squared_error(y_truth, model_predicted_data, squared=False),
        'R-squared': r2_score(y_truth, model_predicted_data),
        'MAE': mean_absolute_error(y_truth, model_predicted_data),
        'MAPE': mean_absolute_percentage_error(y_truth, model_predicted_data),
        'MSLE': None,
        'Explained Variance': explained_variance_score(y_truth, model_predicted_data),
        'Median AE': median_absolute_error(y_truth, model_predicted_data)
    }


    if (model_predicted_data < 0).any():
        print("Skipping MSLE calculation due to negative predictions.")
        negative_predictions_df = pd.DataFrame({'Prediction': model_predicted_data[model_predicted_data < 0],
                                                'True_Label': y_truth[model_predicted_data < 0]})
        negative_predictions_df.to_csv(os.path.join(output_folder, 'negative_predictions.csv'), index=False)
    else:
        metrics_dict['MSLE'] = mean_squared_log_error(y_truth, model_predicted_data)

    if isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet, DecisionTreeRegressor)):
        metrics_df = pd.DataFrame([metrics_dict], index=[0])
    else:
        metrics_df = pd.DataFrame([metrics_dict])


    # Plot predictions vs true labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_truth, y=model_predicted_data)
    sns.lineplot(x=y_truth, y=y_truth, color='red', label='Trend Line')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title(f'{model} - Predictions vs True Labels')
    plt.text(0.05, 0.95, f'R-squared: {metrics_dict["R-squared"]:.2f}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')
    plt.savefig(os.path.join(output_folder, f'predictions_vs_true.png'))
    plt.close()

    # Plot residuals
    residuals = y_truth - model_predicted_data
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{model} - Residuals Distribution')
    plt.text(0.05, 0.95, f'R-squared: {metrics_dict["R-squared"]:.2f}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')
    plt.savefig(os.path.join(output_folder, f'residuals_distribution.png'))
    plt.close()


    # Plot Q-Q plot
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{model} - Q-Q Plot')
    plt.savefig(os.path.join(output_folder, f'qq_plot.png'))
    plt.close()

    # Plot residuals vs fitted values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=model_predicted_data, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model} - Residuals vs Fitted Values')
    plt.savefig(os.path.join(output_folder, f'residuals_vs_fitted.png'))
    plt.close()

    # Prediction interval plot
    plt.figure(figsize=(8, 6))
    pred_interval_low = model_predicted_data - 1.96 * residuals.std()
    pred_interval_high = model_predicted_data + 1.96 * residuals.std()
    sns.scatterplot(x=y_truth, y=model_predicted_data, label='Predicted vs True')
    plt.fill_between(y_truth, pred_interval_low, pred_interval_high, color='gray', alpha=0.2, label='95% Prediction Interval')
    sns.lineplot(x=y_truth, y=y_truth, color='red', label='Trend Line')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title(f'{model} - Predictions with Prediction Interval')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'prediction_interval.png'))
    plt.close()

    metrics_df.to_csv(os.path.join(output_folder, 'model_evaluation_metrics.csv'), index=False)

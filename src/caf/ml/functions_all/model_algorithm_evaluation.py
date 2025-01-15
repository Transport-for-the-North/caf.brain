# -*- coding: utf-8 -*-
"""
Created on: 2/13/2024
Original author: Adil Zaheer
"""
import os

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import pandas as pd
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from caf.ml.old_inputs.cafml_inputs import Models
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

        scores_r2 = cross_val_score(model_instance, x, y, cv=5, scoring="r2", n_jobs=-1)
        scores_mse = -cross_val_score(model_instance, x, y, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

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
               categorical_target,
               y_proba):
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

        # Confusion Matrix Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
        plt.close()

        stats_df = simple_logistic_regression_stats(model, data_used_to_predict, output_folder, target_column, model_predicted_data, y_proba)


        # Feature Importance
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame({
                'feature': data_used_to_predict.drop(columns=[target_column]).columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'feature_importances.png'))
            plt.close()

            feature_importance.to_csv(os.path.join(output_folder, 'feature_importances.csv'),
                                      index=False)

        return

    metrics_dict = {
        'Model': model,
        'RMSE': mean_squared_error(y_truth, model_predicted_data, squared=False),
        'R-squared': r2_score(y_truth, model_predicted_data),
        'MAE': mean_absolute_error(y_truth, model_predicted_data),
        'MAPE': mean_absolute_percentage_error(y_truth, model_predicted_data),
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

    metrics_df.to_csv(os.path.join(output_folder, 'model_evaluation_metrics.csv'), index=False)

    plot_regression_diagnostics(y_truth, model_predicted_data, model, output_folder)

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        plot_feature_importance(model, data_used_to_predict, target_column, output_folder)

    # Coefficients and p-values
    if isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet)):
        calculate_non_categorical_model_stats(model, data_used_to_predict, target_column, output_folder)


def plot_regression_diagnostics(y_truth, model_predicted_data, model, output_folder):

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


def plot_feature_importance(model, data, target_column, output_folder):
    feature_importance = pd.DataFrame({
        'feature': data.drop(columns=[target_column]).columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'feature_importances.png'))
    plt.close()

    feature_importance.to_csv(os.path.join(output_folder, 'feature_importances.csv'), index=False)


def calculate_non_categorical_model_stats(model, data_used_to_predict, target_column, output_folder):
    X = data_used_to_predict.drop(columns=target_column)
    y = data_used_to_predict[target_column]
    feature_names = X.columns.tolist()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # Use permutation importance if feature_importances_ not available
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean

    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    results_df = results_df.sort_values('Importance', ascending=False)
    results_df.to_csv(os.path.join(output_folder, 'non_categorical_model_stats.csv'), index=False)

    return results_df


def simple_logistic_regression_stats(model, data_used_to_predict, output_folder, target_column, model_predicted_data, y_proba):
    X = data_used_to_predict
    pred = model_predicted_data

    if not isinstance(model, LogisticRegression):
        raise ValueError("Model must be an instance of LogisticRegression")

    feature_names = X.columns.tolist()
    results = []

    # binary classification, one set of coefficients
    coef = model.coef_[0]
    intercept = model.intercept_[0]

    # Calculate standard errors
    residuals = pred - y_proba[:, 1]
    mse = np.mean(residuals ** 2)
    std_errors = np.sqrt(mse * (1 / np.sum((X - X.mean()) ** 2, axis=0)))

    # Intercept
    results.append({
        'Feature': 'Intercept',
        'Coefficient': round(intercept, 5),
        'Std_Error': 'N/A',
        'Z_Score': 'N/A',
        'P_Value': 'N/A',
        'Odds_Ratio': round(np.exp(intercept), 5)
    })

    # Features
    for feature, coef_value, std_err in zip(feature_names, coef, std_errors):
        z_score = coef_value / std_err
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        results.append({
            'Feature': feature,
            'Coefficient': round(coef_value, 5),
            'Std_Error': round(std_err, 5),
            'Z_Score': round(z_score, 5),
            'P_Value': round(p_value, 10),
            'Odds_Ratio': round(np.exp(coef_value), 5)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, 'logistic_regression_stats.csv'), index=False)
    return results_df

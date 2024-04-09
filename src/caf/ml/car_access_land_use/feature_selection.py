# -*- coding: utf-8 -*-
"""
Created on: 1/12/2024
Original author: Adil Zaheer
"""
import os

# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, \
    cross_val_score
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from caf.ml.inputs.cafml_inputs import CarAccessInputs, Models, Default_regression_methods, CV_models
from caf.ml.car_access_land_use.process_data_class import process_data_numeric
from caf.ml.functions.model_algorithm_evaluation import select_model
from typing import Union
from sklearn.metrics import mean_squared_error


def feature_selection_cv(data,
                         model_type,
                         cv_method,
                         splits,
                         repeats,
                         target_column):
    # MODEL CHECKS
    if cv_method is not None and cv_method not in CV_models:
        raise ValueError(f"Invalid cross-validation method: {cv_method}")

    valid_models = [model.value for model in Models] + Default_regression_methods
    if model_type is None:
        model_type = Default_regression_methods
    elif not isinstance(model_type, list):  # Ensure model_type is a list
        model_type = [model_type]
    for mod in model_type:
        if mod is not None and mod not in valid_models:
            raise ValueError(
                f"Invalid model type: {mod}. Please provide valid model types from Models class or Default_regression_methods.")

    # Split data into X and y
    x = data.drop(columns=[target_column])
    y = data[target_column]
    scaler = StandardScaler()

    # SELECT REGRESSION METHOD
    if len(model_type) == 1:
        model = model_type[0].value
    elif len(model_type) > 1:
        model = select_model(x, y, model_type)
    else:
        raise ValueError("No valid model provided.")

    cv = get_cv_class(cv_method, splits=splits, repeats=repeats)

    # Initialise dictionaries to store selected features and their performance metrics
    best_features_model = []
    best_features_f_classif = []
    best_features_rfe = []
    best_score_model = float('-inf')
    best_score_f_classif = float('-inf')
    best_score_rfe = float('-inf')

    # Cross-validation loop
    for train_index, test_index in cv.split(x, y):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale the data
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit the model
        model.fit(X_train_scaled, y_train)

        # Feature selection using SelectFromModel
        selected_model = SelectFromModel(model, prefit=True)
        X_train_model = selected_model.transform(X_train_scaled)
        score_model = cross_val_score(model, X_train_model, y_train,
                                      scoring='neg_mean_squared_error', cv=cv).mean()
        if score_model > best_score_model:
            best_score_model = score_model
            best_features_model = X_train.columns[selected_model.get_support()].tolist()

        # Feature selection using f_classif
        selected_f_classif = SelectKBest(f_classif, k='all').fit(X_train_scaled, y_train)
        X_train_f_classif = selected_f_classif.transform(X_train_scaled)
        score_f_classif = cross_val_score(model, X_train_f_classif, y_train,
                                          scoring='neg_mean_squared_error', cv=cv).mean()
        if score_f_classif > best_score_f_classif:
            best_score_f_classif = score_f_classif
            best_features_f_classif = X_train.columns[selected_f_classif.get_support()].tolist()

        # Feature selection using RFE
        selector_rfe = RFE(estimator=model, n_features_to_select=5, step=1)
        selector_rfe.fit(X_train_scaled, y_train)
        X_train_rfe = selector_rfe.transform(X_train_scaled)
        score_rfe = cross_val_score(model, X_train_rfe, y_train, scoring='neg_mean_squared_error',
                                    cv=cv).mean()
        if score_rfe > best_score_rfe:
            best_score_rfe = score_rfe
            best_features_rfe = X_train.columns[selector_rfe.support_].tolist()

    print("Best score for SelectFromModel:", best_score_model)
    print("Best score for f_classif:", best_score_f_classif)
    print("Best score for RFE:", best_score_rfe)

    return best_features_model, best_features_f_classif, best_features_rfe, model


def get_cv_class(cv_method, splits, repeats):
    if cv_method:
        if cv_method.lower() == 'kfold':
            return KFold(n_splits=splits if splits else 5, shuffle=True)
        elif cv_method.lower() == 'stratifiedkfold':
            return StratifiedKFold(n_splits=splits if splits else 5, shuffle=True)
        elif cv_method.lower() == 'repeatedkfold':
            return RepeatedKFold(n_splits=splits if splits else 5, n_repeats=repeats)
        elif cv_method.lower() == 'repeatedstratifiedkfold':
            return RepeatedStratifiedKFold(n_splits=splits if splits else 5, n_repeats=repeats)
        else:
            raise ValueError(f"Invalid cross-validation method: {cv_method}")
    else:
        # Default to KFold with 5 splits and shuffle
        return KFold(n_splits=5, shuffle=True)


def filter_data(original_data, best_features_model, best_features_f_classif, best_features_rfe, output_folder, target_column):
    # Combine all selected features into one list
    all_selected_features = best_features_model + best_features_f_classif + best_features_rfe

    # Count occurrences of each feature
    feature_counts = pd.Series(all_selected_features).value_counts()

    # Filter features that appear at least twice
    selected_columns = feature_counts[feature_counts >= 2].index.tolist()

    # Add the target column to the selected columns
    selected_columns.append(target_column)

    # Filter the original data
    filtered_data = original_data[selected_columns]

    output_filename = 'feature_selection_data.csv'
    output_path = os.path.join(output_folder, output_filename)
    filtered_data.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"Feature selected data exported to: {output_path}")

    return filtered_data

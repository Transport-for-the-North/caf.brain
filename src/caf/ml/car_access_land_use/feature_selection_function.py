# -*- coding: utf-8 -*-
"""
Created on: 12/29/2023
Updated on:

Original author: Adil Zaheer
Last update made by:
Other updates made by:

File purpose:

"""
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold


def feature_selection(data_to_model, target_column=None, regression_methods=None,
                      alpha_range=None, cv_method='kfold', repeats=None, splits=None,
                      num_folds=5, alpha_target=9, selected_algorithm=None,
                      threshold_for_best_alpha=None):
    # Check if all columns are numeric
    if not data_to_model.applymap(np.isreal).all().all():
        print("Warning: Not all columns are numeric. Converting non-numeric columns to numeric.")

        # Convert non-numeric columns to numeric, drop non-convertible columns
        data_to_model = data_to_model.apply(pd.to_numeric, errors='coerce')
        data_to_model = data_to_model.dropna(axis=1, how='any')

    # Check if the target column is numeric
    if target_column is not None and not pd.to_numeric(data_to_model[target_column],
                                                       errors='coerce').notna().all():
        raise ValueError(
            "Error: The target column is not numeric. Please adapt the target column to be numeric.")

    # Remove any rows with NaN values after conversion
    data_to_model = data_to_model.dropna()

    # Extract X and y
    X = data_to_model.drop(target_column, axis=1).values
    y = data_to_model[target_column].values

    # Apply feature scaling
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)

    # Initialize variables
    default_regression_methods = [ElasticNet, Lasso, Ridge]
    regression_methods = regression_methods or default_regression_methods

    if selected_algorithm and selected_algorithm not in regression_methods:
        raise ValueError(f"Selected algorithm {selected_algorithm} not in the list of regression methods.")

    # If a specific algorithm is selected, use only that one
    if selected_algorithm:
        regression_methods = [selected_algorithm]

    # Initialize variables
    best_alpha = float('inf')
    best_score = float('inf')
    final_model = None
    if threshold_for_best_alpha is None:
        threshold_for_best_alpha = 9  # Set a default value if not provided

    # default alpha range to try
    if alpha_range is None:
        alpha_range = np.arange(0.1, 10, 0.1)

    for regression_method in regression_methods:
        # Outer cross-validation loop for model evaluation
        outer_scores = []
        selected_features = None
        X_selected_combined = None

        # Data split into training and test sets with specified cross-validation method
        cv_class = get_cv_class(cv_method, splits, repeats)
        for train_index, test_index in tqdm(cv_class.split(X_scaled),
                                            desc=f"Outer CV Progress - {regression_method.__name__}"):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Inner cross-validation loop for feature selection
            selector = SelectFromModel(estimator=regression_method())
            final_model_inner = regression_method(max_iter=1000)

            # Apply feature selection to training and target data
            selector.fit(X_train, y_train)
            selected_features = selector.get_support()

            # Check if any features are selected
            if not any(selected_features):
                print(
                    f"No features selected for {regression_method.__name__}. Skipping grid search.")
                continue

            X_selected = selector.transform(X_train)

            # In the 1st loop iteration, X_selected_combined is initialized with zeros
            if X_selected_combined is None:
                X_selected_combined = np.zeros((X_scaled.shape[0], X_scaled.shape[1]))
            X_selected_combined[train_index[:, np.newaxis], selected_features] = X_selected

            # Perform a grid search with cross-validation to find the best alpha for selected features
            grid_search = GridSearchCV(estimator=final_model_inner,
                                       param_grid={'alpha': alpha_range},
                                       scoring='neg_mean_squared_error', cv=num_folds)
            grid_search.fit(X_selected, y_train)

            # Get the best alpha and best score from the inner loop
            best_alpha_inner = grid_search.best_params_['alpha']
            best_score_inner = -grid_search.best_score_

            # Fit the final model with the best alpha from the inner loop
            final_model_inner.alpha = best_alpha_inner
            final_model_inner.fit(X_selected, y_train)

            # Evaluate the model on the outer test set
            X_selected_test = selector.transform(X_test)
            outer_score = final_model_inner.score(X_selected_test, y_test)
            outer_scores.append(outer_score)

            # Update the best alpha and score if necessary
            if best_score_inner < best_score:
                best_score = best_score_inner
                best_alpha = best_alpha_inner
                final_model = final_model_inner

        print(f"{regression_method.__name__} - Best Alpha: {best_alpha}, Best Score: {best_score}")

        # Check if best_alpha exceeds the threshold
        if best_alpha is not None and best_alpha >= threshold_for_best_alpha:
            print(
                f"Alpha value is {best_alpha}. Consider increasing alpha range or adjusting the threshold.")

    # Check if no features are selected for any regression method
    if X_selected_combined is None or np.all(X_selected_combined == 0):
        print("No features selected for any regression method")
        final_model = None
        selected_features_df = None

    else:
        print("Selected Features:", np.array(data_to_model.columns[:-1])[selected_features])
        print("Mean Outer Score:", np.mean(outer_scores))

        # Check if a user-defined alpha range is provided
        threshold_for_best_alpha = alpha_target if alpha_range is None else alpha_target

        # Check if best_alpha exceeds the threshold
        if best_alpha >= threshold_for_best_alpha:
            print(
                f"Alpha value is {best_alpha}. Consider increasing alpha range or adjusting the threshold.")

        selected_features_df = pd.DataFrame(X_selected_combined[:, selected_features],
                                            columns=np.array(data_to_model.columns[:-1])[
                                                selected_features])

    return final_model, selected_features_df


def get_cv_class(cv_method, splits, repeats):
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

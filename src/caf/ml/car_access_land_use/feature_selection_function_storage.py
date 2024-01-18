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
        feat_select_df = feature_selection_alt(X_scaled, y)
        # if this feat selection is called here then data already sorted, if just called as user doesnt
        # want the intensive one then need to sort data a bit like the code does in new feat selec func
        # needs adapting this is TODO tomorrow
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


# feature selection for if initial feature selection results in all features removed or if user specified

from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from scipy.stats import ttest_ind


def feature_selection_alt(data, target_column, X_scaled, y, correlation_threshold=0.5, p_value_threshold=0.05, k_best=5):

    data_list = [data, X_scaled, y]
    for i in range(len(data_list)):
        # For the data DataFrame
        if i == 0:
            if target_column in data_list[i].columns:
                try:
                    data_list[i][target_column] = pd.to_numeric(data_list[i][target_column],
                                                                errors='coerce')
                except ValueError:
                    print(
                        f"Error: Unable to convert '{target_column}' to numeric. Please check the target column.")
                    return
            else:
                print(
                    f"Warning: '{target_column}' not found in the 'data' DataFrame. Make sure the target column is specified correctly.")
                return
            if not data_list[i].applymap(np.isreal).all().all():
                print(
                    "Warning: Not all columns are numeric. Converting non-numeric columns to numeric.")
                data_list[i] = data_list[i].apply(pd.to_numeric, errors='coerce')
                data_list[i] = data_list[i].dropna(axis=1, how='any')
                print("Columns dropped:", data_list[i].columns.difference(data.columns))
            # For X_scaled and y
            else:
                if i == 2:  # For y
                    if not data_list[i].applymap(np.isreal).all().all():
                        print(
                            f"Error: All columns in '{data_list[i].name}' should be numeric. Please check your data.")
                        return
                else:  # For X_scaled
                    if not data_list[i].applymap(np.isreal).all().all():
                        print(
                            f"Warning: Not all columns in '{data_list[i].name}' are numeric. Converting non-numeric columns to numeric.")
                        data_list[i] = data_list[i].apply(pd.to_numeric, errors='coerce')
                        data_list[i] = data_list[i].dropna(axis=1, how='any')
                        print("Columns dropped:", data_list[i].columns.difference(data.columns))



    # Remove any rows with NaN values after conversion
    data = data.dropna()

    # Split data into X (features) and y (target variable)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Pairwise Correlation
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]

    # ANOVA (for numerical features)
    numeric_columns = X.select_dtypes(include=np.number).columns
    anova_results = pd.Series(index=numeric_columns)
    for column in numeric_columns:
        _, p_value = f_classif(X[[column]], y)
        anova_results[column] = p_value
    to_drop_anova = anova_results[anova_results > p_value_threshold].index.tolist()

    # T-tests (for binary categorical features)
    binary_columns = X.select_dtypes(include='category').columns
    t_test_results = pd.Series(index=binary_columns)
    for column in binary_columns:
        group1 = X[X[column] == X[column].value_counts().idxmax()][target_column]
        group2 = X[X[column] != X[column].value_counts().idxmax()][target_column]
        _, p_value = ttest_ind(group1, group2)
        t_test_results[column] = p_value
    to_drop_t_test = t_test_results[t_test_results > p_value_threshold].index.tolist()

    # Chi-squared tests (for non-binary categorical features)
    non_binary_columns = X.select_dtypes(include='category').columns.difference(binary_columns)
    chi2_results = pd.Series(index=non_binary_columns)
    for column in non_binary_columns:
        contingency_table = pd.crosstab(X[column], y)
        _, p_value, _, _ = chi2(contingency_table)
        chi2_results[column] = p_value
    to_drop_chi2 = chi2_results[chi2_results > p_value_threshold].index.tolist()

    # Combine features to drop from all tests
    to_drop_all = list(set(to_drop_corr + to_drop_anova + to_drop_t_test + to_drop_chi2))

    # Select top k features using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k_best)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()

    # Print features to drop
    print("Features removed:")
    print(to_drop_all)

    # Create DataFrame with selected features
    selected_data = pd.concat([X[selected_features], y], axis=1)

    return selected_data



def pvalue_feature_selection_(x_train, y_train, x_test):
    significance_level = 0.05
    methods = [f_classif, chi2, mutual_info_classif]

    # Initialize a dictionary to store p-values for each method
    p_values_dict = {}

    # Loop through each statistical test method
    for method in methods:
        # Use SelectKBest with k='all' to compute scores for all features on training data
        selector = SelectKBest(method, k='all')
        selector.fit(x_train, y_train)

        # Get p-values from the statistical test
        p_values = selector.pvalues_

        # Convert p-values to numpy array to ensure consistent shape
        p_values = np.asarray(p_values)

        # Store p-values in the dictionary
        p_values_dict[method.__name__] = p_values

    # Calculate average p-values across all methods
    average_p_values = np.mean(list(p_values_dict.values()), axis=0)

    # Select features that pass the significance level on average for both training and test data
    selected_features_train = x_train.columns[average_p_values < significance_level].tolist()
    selected_features_test = x_test.columns[average_p_values < significance_level].tolist()

    return selected_features_train, selected_features_test
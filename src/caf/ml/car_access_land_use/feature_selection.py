# -*- coding: utf-8 -*-
"""
Created on: 1/12/2024
Updated on:

Original author: Adil Zaheer
Last update made by:
Other updates made by:

File purpose:

"""
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# todo notes to self below, ignore
# training data testing data validation data, vald. data used to reign in the treating data, test dat. used at end. once we've got best combo then try on test data
# instead of gridseach use find random as part of a grid etc. parameter sampler
# leave one out and randomised search
# break up feature selection and hyper param optim. should be able to do hyper param optim without feature selection,
# functionalise in different way.



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression


def feature_selection_(df, target_column, model_type, cv_method=None,
                       splits=None, repeats=None,
                       hp_optimisation=None):

    ######## DATA PROCESSING ########

    # Check if all columns are numeric
    if not df.applymap(np.isreal).all().all():
        print("Not all columns are numeric. Converting non-numeric columns to numeric.")

        # Convert non-numeric columns to numeric, drop non-convertible columns
        df = df.apply(pd.to_numeric, errors='coerce')
        data = df.dropna(axis=1, how='any')
    else:
        data = df  # Initialise data if all columns are numeric

    # Remove any rows with NaN values after conversion
    data_to_model = data.dropna()

    # split data into features and target
    x = data_to_model.drop(target_column, axis=1)
    y = data_to_model[target_column]

    # scale features
    scale = StandardScaler()
    x_scaled = scale.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled)

    ######## ALGORITHM SELECTION ########

    # Initialize regression methods
    default_regression_methods = [ElasticNet, Lasso, Ridge]
    model_classes = {
        "linear_regression": LinearRegression,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "svm": SVR,
        "knr": KNeighborsRegressor,
        "neural_network": MLPRegressor,
        "gradient_boosting": GradientBoostingRegressor,
    }

    if model_type is None:
        regression_methods = default_regression_methods
    else:
        if model_type not in model_classes:
            raise ValueError(
                f"Selected algorithm {model_type} not in the list of regression methods.")
        regression_methods = [model_classes[model_type]]


    cv_class = get_cv_class(cv_method, splits, repeats)

    if hp_optimisation is None:
        hpo = RandomizedSearchCV
        print(
            "A hyperparameter optimization method was not specified. RandomizedSearchCV is used as default and recommended")
    elif hp_optimisation not in (RandomizedSearchCV, GridSearchCV):
        raise ValueError(
            f"Selected hyperparameter optimization method not in the recommended list. "
            "Please use either RandomizedSearchCV or GridSearchCV")

    ######## FEATURE SELECTION CODE ########

    final_selected_features = []

    for regression_method in regression_methods:
        selected_features = []

        cv_class = get_cv_class(cv_method, splits, repeats)
        for train_index, test_index in tqdm(cv_class.split(x_scaled),
                                            desc=f"Feature selection progress - {regression_method.__name__}"):
            x_train, x_test = x_scaled.iloc[train_index], x_scaled.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Feature selection using different methods
            feature1 = feat_select_func(x_train, y_train, x_test, regression_method)
            feature2 = pvalue_feature_selection_(x_train, y_train, x_test)
            feature3 = recursive_feature_elimination_(x_train, y_train, x_test)

            # Combine features based on criteria
            combined_features = combine_features([feature1, feature2, feature3])

            # Append combined features for each fold
            selected_features.append(combined_features)

        # Take the average best features across all folds
        average_selected_features = combine_features(selected_features)

        # Append the average selected features for the current regression method
        final_selected_features.append(average_selected_features)

    final_selected_features_across_methods = combine_features(final_selected_features)
    print('Final selected features:')
    print('-------------------------------------')
    print(final_selected_features_across_methods)

    return final_selected_features_across_methods


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



def feat_select_func(x_train, y_train, x_test, regression_method):
    if not regression_method:
        raise ValueError("Please provide a regression method for feature selection.")

    selector = SelectFromModel(estimator=regression_method())
    selector.fit(x_train, y_train)

    # Get the selected features indices
    selected_feature_indices_train = selector.get_support(indices=True)
    selected_feature_indices_test = selector.get_support(indices=True)

    # Get the names of selected features
    selected_features_train = x_train.columns[selected_feature_indices_train].tolist()
    selected_features_test = x_test.columns[selected_feature_indices_test].tolist()

    return selected_features_train, selected_features_test


def pvalue_feature_selection_(x_train, y_train, x_test):
    significance_level = 0.05
    methods = [f_classif, mutual_info_classif]  #todo,chi2 removed due to scaled data, works with just x not x_scaled

    # Initialise a dictionary to store selected features counts
    feature_counts = {}

    # Loop through each statistical test method
    for method in methods:
        # Use SelectKBest with k='all' to compute scores for all features on training data
        selector = SelectKBest(method, k='all')
        selector.fit(x_train, y_train)

        # Get selected features
        selected_features = x_train.columns[selector.get_support()].tolist()

        # Update feature counts
        for feature in selected_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Select features that appear at least twice
    selected_features_train = [feature for feature, count in feature_counts.items() if count >= 2]
    selected_feature_indices = [x_test.columns.get_loc(feature) for feature in
                                selected_features_train]

    # Index into x_test.columns using the selected feature indices
    selected_features_test = x_test.columns[selected_feature_indices].tolist()

    return selected_features_train, selected_features_test


def recursive_feature_elimination_(x_train, y_train, x_test):
    estimator = LogisticRegression()
    n_features_to_select = x_train.shape[1]

    # Use RFE with logistic regression on training data
    selector = RFE(estimator, n_features_to_select=n_features_to_select)
    X_selected_train = selector.fit_transform(x_train, y_train)

    # Get selected features on both training and test data
    selected_features_train = x_train.columns[selector.support_].tolist()
    selected_features_test = x_test.columns[selector.support_].tolist()

    return selected_features_train, selected_features_test


def combine_features(feature_sets):
    feature_counts = {}

    # Loop through each feature set and feature, counting occurrences
    for feature_set in feature_sets:
        for feature in feature_set:
            # Convert lists to tuples to make it work, unsure why it doesnt work otherwise #todo
            feature_key = tuple(feature) if isinstance(feature, list) else feature
            feature_counts[feature_key] = feature_counts.get(feature_key, 0) + 1

    # Initialise a list to store the features selected based on occurrences
    average_selected_features = []

    # Loop through each feature and its count in the dictionary
    for feature, count in feature_counts.items():
        # If a feature appears at least twice, add it to the selected features
        if count >= 2:
            average_selected_features.append(feature)
        # If no feature appears at least twice, add all features that appear at least once
        elif count == 1 and all(count < 2 for count in feature_counts.values()):
            average_selected_features.append(feature)

    # Return the list of average selected features
    return average_selected_features

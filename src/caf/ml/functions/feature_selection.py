# -*- coding: utf-8 -*-
"""
Created on: 1/12/2024
Original author: Adil Zaheer
"""
import os
import statsmodels.api as sm
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, \
    GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, \
    cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel, f_regression, mutual_info_regression, \
    mutual_info_classif, RFECV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from caf.ml.inputs.cafml_inputs import Models, Default_regression_methods, CV_models, ModelGrids, Models_List_
from caf.ml.functions.model_algorithm_evaluation import select_model
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression


def feature_selection_cv(data,
                         model_type,
                         cv_method,
                         splits,
                         repeats,
                         target_column,
                         output_folder,
                         skip_feature_selection,
                         basic_model,
                         multiple_year_prediction,
                         categorical_data):

    if skip_feature_selection is not None or basic_model is not None:
        # MODEL CHECKS
        if cv_method is not None and cv_method not in CV_models:
            raise ValueError(f"Invalid cross-validation method: {cv_method}")

        valid_models = [model.value for model in Models] + Default_regression_methods + Models_List_
        if model_type is None:
            model_type = Default_regression_methods
        elif not isinstance(model_type, list):
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
            model = model_type[0].get_model_instance() if isinstance(model_type[0], Models) else model_type[0]
            #model = model_type[0].value()
        elif len(model_type) > 1:
            model = select_model(x, y, model_type, output_folder=output_folder)
        else:
            raise ValueError("No valid model provided.")
        print('Model selected:')
        print(model)
        return [], [], [], [], [], model

    # MODEL CHECKS
    if cv_method is not None and cv_method not in CV_models:
        raise ValueError(f"Invalid cross-validation method: {cv_method}")

    valid_models = [model.value for model in Models] + Default_regression_methods + Models_List_
    if model_type is None:
        model_type = Default_regression_methods
    elif not isinstance(model_type, list):
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
        model = model_type[0].get_model_instance() if isinstance(model_type[0], Models) else model_type[0]
        #model = model_type[0].value()
    elif len(model_type) > 1:
        model = select_model(x, y, model_type, output_folder=output_folder)
    else:
        raise ValueError("No valid model provided.")
    print(model)

    if isinstance(model, (RandomForestRegressor, ExtraTreesRegressor)):
        if isinstance(model, DecisionTreeRegressor):
            model.set_params(max_depth=10)
        elif isinstance(model, GradientBoostingRegressor):
            model.set_params(max_depth=5, n_estimators=10)
        else:
            model.set_params(n_estimators=10, n_jobs=-1)
    elif isinstance(model, SVR):
        model.set_params(max_iter=3000, tol=1e-3)

    best_features_model = []
    best_features_f_regressor = []
    best_features_rfe = []
    best_score_model = float('-inf')
    best_score_f_regressor = float('-inf')
    best_score_rfe = float('-inf')

    best_features_sfs = []
    best_features_mutual_info = []
    best_score_mutual_info = float('-inf')
    best_score_sfs = float('-inf')

    cv = TimeSeriesSplit(n_splits=5) if multiple_year_prediction is not None else get_cv_class(cv_method, splits=splits, repeats=repeats)

    if categorical_data is not None:
        for train_index, test_index in tqdm(cv.split(x, y), total=cv.get_n_splits(x, y), desc="Feature selection progress"):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)

            # RFECV for feature selection
            rfecv = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy')
            selector = rfecv.fit(X_train, y_train)
            X_train_rfe = selector.transform(X_train)
            score_rfe = cross_val_score(model, X_train_rfe, y_train, scoring='accuracy', cv=cv).mean()
            if score_rfe > best_score_rfe:
                best_score_rfe = score_rfe
                support_mask_rfe = selector.support_
                best_features_rfe = X_train.columns[support_mask_rfe].to_list()

            # Mutual Information for feature selection
            selector_mi = SelectKBest(mutual_info_classif, k='all').fit(X_train, y_train)
            X_selected_mi = selector_mi.transform(X_train)
            score_f_regressor = cross_val_score(model, X_selected_mi, y_train, scoring='accuracy', cv=cv).mean()
            if score_f_regressor > best_score_f_regressor:
                best_score_f_regressor = score_f_regressor
                support_mask_mi = selector_mi.get_support()
                best_features_f_regressor = X_train.columns[support_mask_mi].to_list()


        if best_score_model != float('-inf'):
            print("Best score for SelectFromModel:", best_score_model)
        if best_score_f_regressor != float('-inf'):
            print("Best score for mutual_info_classif:", best_score_f_regressor)
        if best_score_rfe != float('-inf'):
            print("Best score for RFE:", best_score_rfe)
        print('Trained model', model)

        return best_features_model, best_features_f_regressor, best_features_rfe, best_features_sfs, best_features_mutual_info, model

    else:

        for train_index, test_index in tqdm(cv.split(x, y), total=cv.get_n_splits(x, y), desc="Feature selection progress"):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)

            if isinstance(model, (KNeighborsRegressor, BaggingRegressor, SVR)):
                sfs = SFS(model,
                          k_features=5,
                          forward=True,
                          floating=False,
                          scoring='neg_mean_squared_error',
                          cv=cv,
                          n_jobs=-1)
                sfs.fit(X_train_scaled, y_train)
                score_sfs = sfs.k_score_
                if score_sfs > best_score_sfs:
                    best_score_sfs = score_sfs
                    if sfs.k_feature_idx_ is not None:
                        best_features_sfs = [X_train.columns[int(i)] for i in sfs.k_feature_idx_]
                    else:
                        best_features_sfs = []

                selector_mutual_info = SelectKBest(mutual_info_regression, k=5).fit(X_train_scaled, y_train)
                X_train_mutual_info = selector_mutual_info.transform(X_train_scaled)
                score_mutual_info = cross_val_score(model, X_train_mutual_info, y_train,
                                           scoring='neg_mean_squared_error', cv=cv).mean()
                if score_mutual_info > best_score_mutual_info:
                    best_score_mutual_info = score_mutual_info
                    best_features_mutual_info = X_train.columns[selector_mutual_info.get_support()].tolist()

            else:
                # Feature selection using SelectFromModel
                selected_model = SelectFromModel(model, prefit=True)
                X_train_model = selected_model.transform(X_train_scaled)
                score_model = cross_val_score(model, X_train_model, y_train,
                                              scoring='neg_mean_squared_error', cv=cv).mean()
                if score_model > best_score_model:
                    best_score_model = score_model
                    best_features_model = X_train.columns[selected_model.get_support()].tolist()

                # Feature selection using f_classif
                selected_f_regressor = SelectKBest(f_regression, k='all').fit(X_train_scaled, y_train)
                X_train_f_regressor = selected_f_regressor.transform(X_train_scaled)
                score_f_regressor = cross_val_score(model, X_train_f_regressor, y_train,
                                                  scoring='neg_mean_squared_error', cv=cv).mean()
                if score_f_regressor > best_score_f_regressor:
                    best_score_f_regressor = score_f_regressor
                    best_features_f_regressor = X_train.columns[selected_f_regressor.get_support()].tolist()

                # Feature selection using RFE
                selector_rfe = RFE(estimator=model, n_features_to_select=5, step=1)
                selector_rfe.fit(X_train_scaled, y_train)
                X_train_rfe = selector_rfe.transform(X_train_scaled)
                score_rfe = cross_val_score(model, X_train_rfe, y_train, scoring='neg_mean_squared_error',
                                            cv=cv).mean()
                if score_rfe > best_score_rfe:
                    best_score_rfe = score_rfe
                    best_features_rfe = X_train.columns[selector_rfe.support_].tolist()

        if best_score_model != float('-inf'):
            print("Best score for SelectFromModel:", best_score_model)
        if best_score_f_regressor != float('-inf'):
            print("Best score for f_classif:", best_score_f_regressor)
        if best_score_rfe != float('-inf'):
            print("Best score for RFE:", best_score_rfe)
        if best_score_sfs != float('-inf'):
            print("Best score from Sequential Feature Selection:", best_score_sfs)
        if best_score_mutual_info != float('-inf'):
            print("Best score from Mutual info regression:", best_score_mutual_info)

        return best_features_model, best_features_f_regressor, best_features_rfe, best_features_sfs, best_features_mutual_info, model


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


# todo - look into this. one form of feature selection may be better than a combination
def filter_data(original_data,
                output_folder,
                target_column,
                index_col,
                best_features_model=None,
                best_features_f_regressor=None,
                best_features_rfe=None,
                best_features_sfs=None,
                best_features_mutual_info=None,
                model=None):

    if best_features_model is None:
        best_features_model = []
    if best_features_f_regressor is None:
        best_features_f_regressor = []
    if best_features_rfe is None:
        best_features_rfe = []
    if best_features_sfs is None:
        best_features_sfs = []
    if best_features_mutual_info is None:
        best_features_mutual_info = []


    if isinstance(model, (KNeighborsRegressor, BaggingRegressor, SVR)):
        all_selected_features = best_features_sfs + best_features_mutual_info
        feature_counts = pd.Series(all_selected_features).value_counts()
        selected_columns = feature_counts[feature_counts >= 1].index.tolist()
        selected_columns.append(target_column)
        filtered_data = original_data[selected_columns]

    elif isinstance(model, sm.Probit):
        all_selected_features = best_features_rfe + best_features_mutual_info
        feature_counts = pd.Series(all_selected_features).value_counts()
        selected_columns = feature_counts[feature_counts >= 1].index.tolist()
        selected_columns.append(target_column)
        filtered_data = original_data[selected_columns]

    elif isinstance(model, LogisticRegression):
        all_selected_features = best_features_rfe + best_features_f_regressor
        feature_counts = pd.Series(all_selected_features).value_counts()
        selected_columns = feature_counts[feature_counts >= 1].index.tolist()
        selected_columns.append(target_column)
        filtered_data = original_data[selected_columns]

    else:
        all_selected_features = best_features_model + best_features_f_regressor + best_features_rfe
        feature_counts = pd.Series(all_selected_features).value_counts()
        selected_columns = feature_counts[feature_counts >= 2].index.tolist()
        selected_columns.append(target_column)
        filtered_data = original_data[selected_columns]

    output_filename = 'feature_selection_data.csv'
    output_path = os.path.join(output_folder, output_filename)
    filtered_data.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"Feature selected data exported to: {output_path}")


    print("feature_selection_data:")
    print(filtered_data.shape)
    print(filtered_data)

    return filtered_data


def identify_feature_types(df, threshold=10):
    categorical_features = []
    numerical_features = []
    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values <= threshold:
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    return categorical_features, numerical_features

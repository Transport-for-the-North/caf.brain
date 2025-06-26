# -*- coding: utf-8 -*-
"""
Created on: 1/23/2025
Original author: Adil Zaheer
"""

import numpy as np

# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    RFE,
    mutual_info_regression,
    f_regression,
    RFECV,
    mutual_info_classif,
)
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from tqdm import tqdm
from caf.ml.feature_selection.feature_selection_functions import get_cv_class
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def feature_selection_test(
    train_transformed, cv, model, weight_column, target_column, binary_prediction
):
    # cv set up
    if isinstance(model, LogisticRegression):
        model.set_params(max_iter=1000)
    cv = get_cv_class(cv_method=cv, splits=None, repeats=None)

    x = train_transformed.drop(
        columns=[target_column] + ([weight_column] if weight_column else [])
    )
    y = train_transformed[target_column]
    weight = train_transformed[weight_column].values.flatten() if weight_column else None
    weight_df = train_transformed[weight_column] if weight_column else None
    print(train_transformed[target_column].dtype)
    print(train_transformed[target_column].unique())
    print(y)

    is_classification = len(np.unique(train_transformed[target_column])) <= 2
    # scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
    if is_classification:
        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            scorer = "roc_auc"
        else:
            scorer = "accuracy"
    else:
        scorer = "neg_mean_squared_error"

    scoring = get_scorer(scorer)

    best_features_model = []
    best_features_f_regressor = []
    best_features_rfe = []
    best_score_model = float("-inf")
    best_score_f_regressor = float("-inf")
    best_score_rfe = float("-inf")

    best_features_sfs = []
    best_features_mutual_info = []
    best_score_mutual_info = float("-inf")
    best_score_sfs = float("-inf")

    if binary_prediction is not None:
        for train_index, test_index in tqdm(
            cv.split(x), total=cv.get_n_splits(x), desc="Feature selection progress"
        ):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            print("Train unique values:", np.unique(y.iloc[train_index]))
            print("Test unique values:", np.unique(y.iloc[test_index]))

            print(y_train)
            print(y_test)
            model.fit(X_train, y_train)

            # RFECV for feature selection
            rfecv = RFECV(estimator=model, step=1, cv=cv, scoring=scoring)
            selector = rfecv.fit(X_train, y_train)
            X_train_rfe = selector.transform(X_train)
            score_rfe = cross_val_score(
                model, X_train_rfe, y_train, scoring=scoring, cv=cv
            ).mean()
            if score_rfe > best_score_rfe:
                best_score_rfe = score_rfe
                support_mask_rfe = selector.support_
                best_features_rfe = X_train.columns[support_mask_rfe].to_list()

            # Mutual Information for feature selection
            selector_mi = SelectKBest(mutual_info_classif, k="all").fit(X_train, y_train)
            X_selected_mi = selector_mi.transform(X_train)
            score_f_regressor = cross_val_score(
                model, X_selected_mi, y_train, scoring=scoring, cv=cv
            ).mean()
            if score_f_regressor > best_score_f_regressor:
                best_score_f_regressor = score_f_regressor
                support_mask_mi = selector_mi.get_support()
                best_features_f_regressor = X_train.columns[support_mask_mi].to_list()

        if best_score_model != float("-inf"):
            print("Best score for SelectFromModel:", best_score_model)
        if best_score_f_regressor != float("-inf"):
            print("Best score for mutual_info_classif:", best_score_f_regressor)
        if best_score_rfe != float("-inf"):
            print("Best score for RFE:", best_score_rfe)

        final_data = filter_data(
            original_data=train_transformed,
            target_column=target_column,
            best_features_model=best_features_model,
            best_features_f_regressor=best_features_f_regressor,
            best_features_rfe=best_features_rfe,
            best_features_sfs=best_features_sfs,
            best_features_mutual_info=best_features_mutual_info,
            model=model,
        )

        return final_data

    else:
        for train_index, test_index in tqdm(
            cv.split(x), total=cv.get_n_splits(x), desc="Feature selection progress"
        ):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)

            if isinstance(model, (KNeighborsRegressor, BaggingRegressor, SVR)):
                sfs = SFS(
                    model,
                    k_features=5,
                    forward=True,
                    floating=False,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                )
                sfs.fit(X_train, y_train)
                score_sfs = sfs.k_score_
                if score_sfs > best_score_sfs:
                    best_score_sfs = score_sfs
                    if sfs.k_feature_idx_ is not None:
                        best_features_sfs = [
                            X_train.columns[int(i)] for i in sfs.k_feature_idx_
                        ]
                    else:
                        best_features_sfs = []

                selector_mutual_info = SelectKBest(mutual_info_regression, k=5).fit(
                    X_train, y_train
                )
                X_train_mutual_info = selector_mutual_info.transform(X_train)
                score_mutual_info = cross_val_score(
                    model, X_train_mutual_info, y_train, scoring=scoring, cv=cv
                ).mean()
                if score_mutual_info > best_score_mutual_info:
                    best_score_mutual_info = score_mutual_info
                    best_features_mutual_info = X_train.columns[
                        selector_mutual_info.get_support()
                    ].tolist()

            else:
                # Feature selection using SelectFromModel
                selected_model = SelectFromModel(model, prefit=True)
                X_train_model = selected_model.transform(X_train)
                score_model = cross_val_score(
                    model, X_train_model, y_train, scoring=scoring, cv=cv
                ).mean()
                if score_model > best_score_model:
                    best_score_model = score_model
                    best_features_model = X_train.columns[
                        selected_model.get_support()
                    ].tolist()

                # Feature selection using f_classif
                selected_f_regressor = SelectKBest(f_regression, k="all").fit(X_train, y_train)
                X_train_f_regressor = selected_f_regressor.transform(X_train)
                score_f_regressor = cross_val_score(
                    model, X_train_f_regressor, y_train, scoring=scoring, cv=cv
                ).mean()
                if score_f_regressor > best_score_f_regressor:
                    best_score_f_regressor = score_f_regressor
                    best_features_f_regressor = X_train.columns[
                        selected_f_regressor.get_support()
                    ].tolist()

                # Feature selection using RFE
                selector_rfe = RFE(estimator=model, n_features_to_select=5, step=1)
                selector_rfe.fit(X_train, y_train)
                X_train_rfe = selector_rfe.transform(X_train)
                score_rfe = cross_val_score(
                    model, X_train_rfe, y_train, scoring=scoring, cv=cv
                ).mean()
                if score_rfe > best_score_rfe:
                    best_score_rfe = score_rfe
                    best_features_rfe = X_train.columns[selector_rfe.support_].tolist()

        if best_score_model != float("-inf"):
            print("Best score for SelectFromModel:", best_score_model)
        if best_score_f_regressor != float("-inf"):
            print("Best score for f_classif:", best_score_f_regressor)
        if best_score_rfe != float("-inf"):
            print("Best score for RFE:", best_score_rfe)
        if best_score_sfs != float("-inf"):
            print("Best score from Sequential Feature Selection:", best_score_sfs)
        if best_score_mutual_info != float("-inf"):
            print("Best score from Mutual info regression:", best_score_mutual_info)

        final_data = filter_data(
            original_data=train_transformed,
            target_column=target_column,
            best_features_model=best_features_model,
            best_features_f_regressor=best_features_f_regressor,
            best_features_rfe=best_features_rfe,
            best_features_sfs=best_features_sfs,
            best_features_mutual_info=best_features_mutual_info,
            model=model,
        )

        return final_data


def filter_data(
    original_data,
    target_column,
    best_features_model=None,
    best_features_f_regressor=None,
    best_features_rfe=None,
    best_features_sfs=None,
    best_features_mutual_info=None,
    model=None,
):

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

    elif isinstance(model, LogisticRegression):
        all_selected_features = best_features_rfe + best_features_f_regressor
        feature_counts = pd.Series(all_selected_features).value_counts()
        selected_columns = feature_counts[feature_counts >= 1].index.tolist()
        selected_columns.append(target_column)
        filtered_data = original_data[selected_columns]

    else:
        all_selected_features = (
            best_features_model + best_features_f_regressor + best_features_rfe
        )
        feature_counts = pd.Series(all_selected_features).value_counts()
        selected_columns = feature_counts[feature_counts >= 2].index.tolist()
        selected_columns.append(target_column)
        filtered_data = original_data[selected_columns]

    return filtered_data

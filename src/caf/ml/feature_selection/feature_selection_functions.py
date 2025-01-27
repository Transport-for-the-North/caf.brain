# -*- coding: utf-8 -*-
"""
Created on: 1/21/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (LogisticRegression,
                                  Ridge,
                                  Lasso)
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.model_selection import (cross_val_score,
                                     KFold,
                                     StratifiedKFold,
                                     RepeatedKFold,
                                     RepeatedStratifiedKFold, TimeSeriesSplit)
from tqdm import tqdm
import seaborn as sb


def rf_feature_selection(data,
                         target_column,
                         cv,
                         regression_method,
                         weight_column,
                         binary_prediction):

    if isinstance(regression_method, LogisticRegression):
        regression_method.set_params(max_iter=1000)
    cv = get_cv_class(cv_method=cv, splits=None, repeats=None)

    X = data.drop(columns=[target_column] + ([weight_column] if weight_column else []))
    y = data[target_column]
    weight = data[weight_column].values.flatten() if weight_column else None
    weight_df = data[weight_column] if weight_column else None

    if binary_prediction:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    with tqdm(total=1, desc="Fitting Random Forest") as pbar:
        model.fit(X, y, sample_weight=weight)
        pbar.update(1)

    selector = SelectFromModel(model, prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()

    n_splits = cv.n_splits if hasattr(cv, 'n_splits') else cv
    scoring = (make_scorer(accuracy_score) if len(pd.unique(y)) > 3
               else make_scorer(roc_auc_score))

    fold_scores = []
    with tqdm(total=n_splits, desc="Cross-validation") as pbar:
        for fold in range(n_splits):
            score = cross_val_score(regression_method,
                                    X[selected_features],
                                    y,
                                    cv=cv,
                                    scoring=scoring,
                                    n_jobs=-1,
                                    verbose=0)[0]
            fold_scores.append(score)
            pbar.update(1)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"Number of features selected: {len(selected_features)}")
    print(f"Cross-validated ROC AUC score: {mean_score:.3f} (+/- {std_score:.3f})")

    if mean_score < 0.6:
        print('Initial feature selection attempt was inaccurate, trying \
              alternative method')
        dataframe_final = feature_selection_intensive(x=X,
                                                      y=y,
                                                      cv=cv,
                                                      regression_method=regression_method,
                                                      weight=weight,
                                                      weight_df=weight_df,
                                                      binary_prediction=binary_prediction)
    else:
        dataframe_final = pd.concat([X[selected_features], y], axis=1)
        dataframe_final = pd.concat([dataframe_final, weight_df], axis=1)
    return dataframe_final


def feature_selection_intensive(x,
                                y,
                                cv,
                                regression_method,
                                weight,
                                weight_df,
                                binary_prediction):

    original_index = x.index

    if binary_prediction:
        selected_features = _classification_feature_selection(x, y, weight)
    else:
        selected_features = _regression_feature_selection(x, y, weight)

    scores = []
    with tqdm(total=cv.n_splits, desc="Cross-validation") as pbar:
        for train_index, test_index in cv.split(x):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            regression_method.fit(X_train[selected_features], y_train)
            score = regression_method.score(X_test[selected_features], y_test)
            scores.append(score)
            pbar.update(1)

    print(f"Number of features selected: {len(selected_features)}")
    print(f"Cross-validated score: {np.mean(scores)}")

    cv_score = np.mean(scores)
    if cv_score < 0.5:
        print('CV score is still not optimal, feature selection is being ignored')
        result = pd.concat([x, y], axis=1)
        if weight_df is not None:
            result = pd.concat([result, weight_df], axis=1)
        return result.set_index(original_index)

    result = pd.concat([x[selected_features], y], axis=1)
    if weight_df is not None:
        result = pd.concat([result, weight_df], axis=1)
    return result.set_index(original_index)


def _classification_feature_selection(x, y, weight):
    rfe = RFE(
        estimator=LogisticRegression(random_state=42, max_iter=2000),
        n_features_to_select=10
    )
    rfe.fit(x, y, sample_weight=weight)
    rfe_selected = x.columns[rfe.support_].tolist()

    logit_lasso = LogisticRegression(penalty='l1', solver='saga', random_state=42)
    logit_lasso.fit(x, y, sample_weight=weight)
    l1_selected = x.columns[abs(logit_lasso.coef_[0]) > 0].tolist()

    return list(set(rfe_selected + l1_selected))


def _regression_feature_selection(x, y, weight):
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(x, y, sample_weight=weight)
    lasso_selected = x.columns[abs(lasso.coef_) > 0].tolist()

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x, y, sample_weight=weight)
    ridge_selected = x.columns[abs(ridge.coef_) > np.mean(abs(ridge.coef_))].tolist()

    return list(set(lasso_selected + ridge_selected))


def get_cv_class(cv_method, splits, repeats):
    if cv_method:
        if cv_method.lower() == 'kfold':
            return KFold(n_splits=splits if splits else 5, shuffle=True)
        elif cv_method.lower() == 'stratifiedkfold':
            return StratifiedKFold(n_splits=splits if splits else 5, shuffle=True)
        elif cv_method.lower() == 'repeatedkfold':
            return RepeatedKFold(n_splits=splits if splits else 5, n_repeats=repeats if repeats else 5)
        elif cv_method.lower() == 'repeatedstratifiedkfold':
            return RepeatedStratifiedKFold(n_splits=splits if splits else 5, n_repeats=repeats if repeats else 5)
        elif cv_method.lower() == 'timeseriessplit':
            return TimeSeriesSplit(n_splits=splits if splits else 5)
        else:
            raise ValueError(f"Invalid cross-validation method: {cv_method}")
    else:
        return KFold(n_splits=5, shuffle=True)


def analyse_feature_importance(train_transformed: pd.DataFrame,
                               target_column: str,
                               weight_column: str,
                               output_path: str) -> pd.DataFrame:
    pd.set_option('display.float_format', lambda x: '%.10f' % x)

    X = train_transformed.drop(columns=[target_column] + ([weight_column] if weight_column else []))
    y = train_transformed[target_column]


    is_classification = len(np.unique(train_transformed[target_column])) <= 2
    if is_classification:
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    else:
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    rf.fit(X, y)
    results = {}

    # Random Forest importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_rf': rf.feature_importances_
    }).sort_values('importance_rf', ascending=False)
    results['random_forest_importance'] = importance_df

    # Permutation importance
    perm_importance = permutation_importance(
        rf, X, y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    perm_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean_perm': perm_importance.importances_mean,
        'importance_std_perm': perm_importance.importances_std
    }).sort_values('importance_mean_perm', ascending=False)
    results['permutation_importance'] = perm_importance_df

    # Target correlations
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [abs(X[col].corr(y)) for col in X.columns]
    }).sort_values('correlation', ascending=False)
    results['target_correlations'] = correlations

    results_df = pd.concat([
        importance_df.set_index('feature'),
        perm_importance_df.set_index('feature')[['importance_mean_perm', 'importance_std_perm']],
        correlations.set_index('feature')
    ], axis=1)

    results_df.to_csv(os.path.join(output_path, 'feature_importances.csv'), float_format='%.10f')

    importance_metrics, filtered_data = filtering_results(results_df=results_df,
                                                          target_column=target_column,
                                                          weight_column=weight_column,
                                                          original_data=train_transformed)

    create_importance_plots(results_df=importance_metrics,
                            output_path=output_path)

    return filtered_data


def filtering_results(results_df,
                      target_column,
                      weight_column,
                      original_data):

    important_features = results_df[
        (results_df['importance_rf'] > 0.05) |
        (results_df['correlation'] > 0.3) |
        (results_df['importance_mean_perm'] > 0.01)
        ]

    selected_features = important_features.index.tolist()
    additional_columns = []
    if target_column:
        additional_columns.append(target_column)
    if weight_column:
        additional_columns.append(weight_column)

    return important_features, original_data[selected_features + additional_columns]


def combine_results(train_final,
                    target_column,
                    weight_column,
                    test):
    if target_column in train_final.columns and weight_column in train_final.columns:
        df = train_final.drop(columns=[target_column, weight_column])
    elif target_column in train_final.columns:
        df = train_final.drop(columns=[target_column])
    elif weight_column in train_final.columns:
        df = train_final.drop(columns=[weight_column])
    else:
        df = train_final

    test_final = test[df.columns]

    return test_final


def create_importance_plots(results_df: pd.DataFrame,
                            output_path: str) -> None:
    # rf importance
    plt.figure(figsize=(12, 6))
    sb.barplot(
        data=results_df.reset_index().head(10),
        x='importance_rf',
        y='feature'
    )
    plt.title('Top 10 Features by Random Forest Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'rf_importance.png'))
    plt.close()

    # Permutation importance
    plt.figure(figsize=(12, 6))
    results_plot = results_df.reset_index().head(10)
    sb.barplot(
        data=results_plot,
        x='importance_mean_perm',
        y='feature',
        xerr=results_plot['importance_std_perm']
    )
    plt.title('Top 10 Features by Permutation Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'perm_importance.png'))
    plt.close()

    # correlations
    plt.figure(figsize=(12, 6))
    sb.barplot(
        data=results_df.reset_index().head(10),
        x='correlation',
        y='feature'
    )
    plt.title('Top 10 Features by Correlation with Target')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'correlations.png'))
    plt.close()

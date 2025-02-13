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
import logging
LOG = logging.getLogger(__name__)

def rf_feature_selection(data: pd.DataFrame,
                         target_column: str,
                         cv: str,
                         regression_method,
                         weight_column: str,
                         classification_prediction: bool,
                         is_time_series: bool) -> pd.DataFrame:
    """
    Two stage feature selection through Random Forest importance and if
    required, a combination of algorithms.

    :param data: Transformed input data split into training set.
    :param target_column: String column name of value to predict.
    :param cv: Cross validation method passed as a string. Any popular
               SciKitlearn methods are suitable with KFold being default if
               left as None.
    :param regression_method: Initialised model algorithm from Models enum class.
    :param weight_column: Optional string column value to be used as weight.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.
    :param is_time_series: If true then data must be time series. Time series
                           based characteristics are taken into consideration
                           during function execution.

    :return:
        dataframe_final: Training data post feature selection.
    """

    if isinstance(regression_method, LogisticRegression):
        regression_method.set_params(max_iter=1000)
    cv = get_cv_class(cv_method=cv, splits=None, repeats=None, is_time_series=is_time_series)

    X = data.drop(columns=[target_column] + ([weight_column] if weight_column else []))
    y = data[target_column]
    weight = data[weight_column].values.flatten() if weight_column else None
    weight_df = data[weight_column] if weight_column else None

    if classification_prediction:
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

    LOG.info(f"Number of features selected: {len(selected_features)}")
    LOG.info(f"Cross-validated ROC AUC score: {mean_score:.3f} (+/- {std_score:.3f})")

    if mean_score < 0.6:
        LOG.warning('Initial feature selection attempt was inaccurate, trying \
              alternative method')
        dataframe_final = feature_selection_intensive(x=X,
                                                      y=y,
                                                      cv=cv,
                                                      regression_method=regression_method,
                                                      weight=weight,
                                                      weight_df=weight_df,
                                                      classification_prediction=classification_prediction)
    else:
        dataframe_final = pd.concat([X[selected_features], y], axis=1)
        dataframe_final = pd.concat([dataframe_final, weight_df], axis=1)
    return dataframe_final


def feature_selection_intensive(x: pd.DataFrame,
                                y: pd.DataFrame,
                                cv: str,
                                regression_method,
                                weight: pd.Series,
                                weight_df: pd.DataFrame,
                                classification_prediction: tuple[int, ...]
                                ) -> pd.DataFrame:
    """
    Thorough feature selection with multiple algorithms.

    :param x: Training data split into explanatory variables only.
    :param y: Training data split only into the target variable.
    :param cv: Cross validation method passed as a string. Any popular
               SciKitlearn methods are suitable with KFold being default if
               left as None.
    :param regression_method: Initialised model algorithm from Models enum class.
    :param weight: Weight values in series form.
    :param weight_df: Weight values in a dataframe.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.
    :return:
        result: Training data post feature selection.
    """
    original_index = x.index

    if classification_prediction:
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

    LOG.info(f"Number of features selected: {len(selected_features)}")
    LOG.info(f"Cross-validated score: {np.mean(scores)}")

    cv_score = np.mean(scores)
    if cv_score < 0.5:
        LOG.warning('CV score is still not optimal, feature selection is being ignored')
        result = pd.concat([x, y], axis=1)
        if weight_df is not None:
            result = pd.concat([result, weight_df], axis=1)
        return result.set_index(original_index)

    result = pd.concat([x[selected_features], y], axis=1)
    if weight_df is not None:
        result = pd.concat([result, weight_df], axis=1)
    return result.set_index(original_index)


def _classification_feature_selection(x: pd.DataFrame,
                                      y: pd.DataFrame,
                                      weight: pd.Series):
    """
    Feature selection algorithms for classification problems.

    :param x: Training data split into explanatory variables only.
    :param y: Training data split only into the target variable.
    :param weight: Weight values in series form.

    :return:
        List of selected features based on both algorithms used.
    """
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


def _regression_feature_selection(x: pd.DataFrame,
                                  y: pd.DataFrame,
                                  weight: pd.Series):
    """
    Feature selection algorithms for regression problems.

    :param x: Training data split into explanatory variables only.
    :param y: Training data split only into the target variable.
    :param weight: Weight values in series form.

    :return:
        List of selected features based on both algorithms used.
    """
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(x, y, sample_weight=weight)
    lasso_selected = x.columns[abs(lasso.coef_) > 0].tolist()

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x, y, sample_weight=weight)
    ridge_selected = x.columns[abs(ridge.coef_) > np.mean(abs(ridge.coef_))].tolist()

    return list(set(lasso_selected + ridge_selected))


def get_cv_class(cv_method: str,
                 splits: int,
                 repeats: int,
                 is_time_series: bool):
    """
    Select which SciKitLearn cross validation method to use.

    :param cv_method: Cross validation method passed as a string. Any popular
                      SciKitlearn methods are suitable with KFold being default if
                      left as None.
    :param splits: Number of splits to be used for cross validation.
    :param repeats: Number of repeats to be used for cross validation.
    :param is_time_series: If true then data must be time series. Time series
                           based characteristics are taken into consideration
                           during function execution.

    :return:
        Initialised cross validation method.
    """
    if is_time_series is True:
        return TimeSeriesSplit(n_splits=splits if splits else 5)
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
            LOG.error(f"Invalid cross-validation method: {cv_method}")
            raise ValueError(f"Invalid cross-validation method: {cv_method}")
    else:
        return KFold(n_splits=5, shuffle=True)


def analyse_feature_importance(train_transformed: pd.DataFrame,
                               target_column: str,
                               weight_column: str,
                               output_path: str) -> pd.DataFrame:
    """
    Simple feature selection through importance and correlation metrics with
    results plotted.

    :param train_transformed: Transformed input data split into training set.
    :param target_column: String column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param output_path: Path to output location.

    :return:
        filtered_data: Training data post feature selection.
    """
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


def filtering_results(results_df: pd.DataFrame,
                      target_column: str,
                      weight_column: pd.DataFrame,
                      original_data: pd.DataFrame):
    """
    Helper function for analyse_feature_importance. Results are analysed and
    applied to input data.

    :param results_df: Dataframe of feature selection scores.
    :param target_column: String column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param original_data: Transformed input data split into training set.

    :return:
        important_features: Feature selection score results.
        original_data: Training data post feature selection.
    """

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


def combine_results(train_final: pd.DataFrame,
                    target_column: str,
                    weight_column: str,
                    test: pd.DataFrame):
    """
    Function to apply feature selection results to training data.

    :param train_final: Final training data post feature selection.
    :param target_column: String column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param test: Transformed input data split into training set.

    :return:
        test_final: Final test data post feature selection.
        cols_dropped_by_feat_select: Dataframe of explanatory variables
                                     removed during feature selection.
    """
    if target_column in train_final.columns and weight_column in train_final.columns:
        df = train_final.drop(columns=[target_column, weight_column])
    elif target_column in train_final.columns:
        df = train_final.drop(columns=[target_column])
    elif weight_column in train_final.columns:
        df = train_final.drop(columns=[weight_column])
    else:
        df = train_final

    test_final = test[df.columns]

    extra_columns = test.columns.difference(df.columns)
    cols_dropped_by_feat_select = test[extra_columns]

    return test_final, cols_dropped_by_feat_select


def create_importance_plots(results_df: pd.DataFrame,
                            output_path: str) -> None:
    """
    Plotting feature selection scores where applicable.

    :param results_df: Feature selection score results.
    :param output_path: Path to output location.

    :return: None
    """
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

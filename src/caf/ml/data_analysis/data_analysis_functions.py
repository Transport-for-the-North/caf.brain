# -*- coding: utf-8 -*-
"""
Created on: 1/17/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os.path
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from caf.ml.process_data_functions.encode_and_scale import preprocess_numerical_data
from sklearn.linear_model import (Ridge,
                                  Lasso,
                                  ElasticNet,
                                  LinearRegression,
                                  LogisticRegression)
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier,
                              ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
import logging
LOG = logging.getLogger(__name__)

def pre_forecast_data_analysis(residuals: pd.Series,
                               model,
                               model_initialised,
                               x_test: pd.Series,
                               train_scaled: pd.DataFrame,
                               test_scaled: pd.DataFrame,
                               full_transformations: bool,
                               train_unscaled: pd.DataFrame,
                               test_unscaled: pd.DataFrame,
                               numerical_features: List[str],
                               categorical_features: List[str],
                               target_column: str,
                               weight_column: str,
                               x_train: pd.Series,
                               output_folder: Path,
                               is_time_series: bool):
    """
    Function to conduct basic data analysis and fix where and if
    applicable.

    :param residuals: Truth values form the train_test_split against the predictions.
    :param model: Fitted model on train_test_split test data.
    :param model_initialised: Initialised SciKitLearn model.
    :param x_test: Series of test data to be used as unseen test data.
    :param train_scaled: Processed input data split into train set.
    :param test_scaled: Processed input data split into train set.
    :param full_transformations: If true then transformations will be applied
                                 to the continuous data inside of train and
                                 test scaled. This fixes any potential data
                                 issues present.
    :param train_unscaled: Input data unprocessed split into train.
    :param test_unscaled: Input data unprocessed split into test.
    :param numerical_features: List of string column names that are
                               continuous variables.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :param target_column: String column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param x_train: Series of train data to be used as train.
    :param output_folder: Path to output location.
    :param is_time_series: If true then data must be time series. Time series
                           based characteristics are taken into consideration
                           during function execution.

    :return:
        train_final: Final train data post data transformations.
        test_final: Final test data post data transformations.
        train_scaled: If transformations are not permitted, train scaled is
                      returned.
        test_scaled: If transformations are not permitted, test scaled is
                      returned.
    """

    alpha = 0.05
    issues = {
        'linearity': False,
        'normality': False,
        'multicolinearity': False,
        'autocorrelation': False,
        'heteroscedasticity': False
    }

    is_statsmodel = any(base.__module__.startswith('statsmodels')
                        for base in model_initialised.__class__.__mro__)
    is_classification = isinstance(model, (LogisticRegression,
                                           GradientBoostingClassifier,
                                           RandomForestClassifier,
                                           ExtraTreesClassifier,
                                           DecisionTreeClassifier,
                                           OneVsRestClassifier)) or (is_statsmodel and
                                           ('Logit' in str(model.__class__) or 'MNLogit' in str(model.__class__)))

    is_linear_model = (isinstance(model, (LinearRegression,
                                          Ridge,
                                          Lasso,
                                          ElasticNet,
                                          LogisticRegression)) or
                       (is_statsmodel and any(
                           name in str(model.__class__) for name in ['OLS', 'GLM', 'Logit', 'MNLogit'])))

    # if is_statsmodel:
    #     X_with_const = add_constant(x_test)
    #     model_for_tests = model
    # else:
    #     # scikit models, convert to stats
    #     X_with_const_train = add_constant(x_train)
    #     if is_classification:
    #         # classification
    #         model_for_tests = sm.Logit(y_train, X_with_const_train).fit(disp=0)
    #     else:
    #         # regression
    #         model_for_tests = sm.OLS(y_train, X_with_const_train).fit()
    #     residuals = model_for_tests.resid
    #     X_with_const = add_constant(x_test)

    # multicolinearity
    LOG.info('Checking for multicollinearity')
    vif_data = pd.DataFrame()
    vif_data["Feature"] = train_scaled.columns
    vif_data["VIF"] = [variance_inflation_factor(train_scaled.values, i) for i in
                       range(train_scaled.shape[1])]
    high_vif = vif_data[vif_data["VIF"] > 10]
    if not high_vif.empty:
        LOG.warning(f"High VIF variables: {high_vif}")
        issues['multicolinearity'] = True

    X_with_const = add_constant(x_test)
    # Breusch-Pagan Heteroscedasticity
    LOG.info('Checking for heteroscedasticity')
    bp_test_statistic, bp_test_p_value, _, _ = het_breuschpagan(residuals, X_with_const)
    LOG.info(f"Breusch-Pagan test p-value: {bp_test_p_value}")

    # White Test Heteroscedasticity
    white_test_statistic, white_test_p_value, _, _ = het_white(residuals, X_with_const)
    LOG.info(f"White's test p-value: {white_test_p_value}")

    if bp_test_p_value < alpha or white_test_p_value < alpha:
        LOG.warning("Warning: Heteroscedasticity detected.")
        issues['heteroscedasticity'] = True


    if is_linear_model and not is_classification:
        LOG.info("Running tests for linear model assumptions")

        # Linearity
        LOG.info('Checking linearity')
        for col in x_train.columns:
            correlation = np.corrcoef(x_train[col], residuals)[0, 1]
            if abs(correlation) > 0.1:
                LOG.warning(f"Warning: {col} may not be linearly related to the target.")
                issues['linearity'] = True

        # Normality
        shapiro_statistic, shapiro_p_value = shapiro(residuals)
        LOG.info(f"Shapiro-Wilk test p-value: {shapiro_p_value}")
        if shapiro_p_value < alpha:
            LOG.warning("Warning: Shapiro-Wilk test suggests non-normality of residuals.")
            issues['normality'] = True

    if is_time_series is not False:
        # Autocorrelation
        dw_statistic = durbin_watson(residuals)
        LOG.info(f"Durbin-Watson statistic: {dw_statistic}")
        if dw_statistic < 1.5 or dw_statistic > 2.5:
            LOG.warning("Warning: Potential autocorrelation in residuals.")
            issues['autocorrelation'] = True

    issues_df = pd.DataFrame(list(issues.items()), columns=['test', 'result'])
    if any(issues.values()) and full_transformations:
        LOG.warning('Data issue present, corrective transformations applied to numerical features')
        if numerical_features is not None:
            train_final = transform_data(df=train_unscaled,
                                         numerical_features=numerical_features,
                                         categorical_features=categorical_features,
                                         target_column=target_column,
                                         weight_column=weight_column)

            test_final = transform_data(df=test_unscaled,
                                        numerical_features=numerical_features,
                                        categorical_features=categorical_features,
                                        target_column=target_column,
                                        weight_column=weight_column)
            issues_df.to_csv(os.path.join(output_folder, 'data_issues_present.csv'))
            return train_final, test_final
        else:
            issues_df.to_csv(os.path.join(output_folder, 'data_issues_present.csv'))
            return train_scaled, test_scaled

    elif any(issues.values()):
        LOG.warning('Data issue present but transformations are not permitted by the user.')
        issues_df.to_csv(os.path.join(output_folder, 'data_issues_present.csv'))
        return train_scaled, test_scaled
    else:
        LOG.info('No data issues present.')
        return train_scaled, test_scaled


def transform_data(df: pd.DataFrame,
                   numerical_features: List[str],
                   categorical_features: List[str],
                   target_column: str,
                   weight_column: str):
    """
    Function to apply data transformations.

    :param df: Processed input data split into train or test set.
    :param numerical_features: List of string column names that are
                               continuous variables.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :param target_column: String column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.

    :return:
        transformed_df: Transformed data.
    """
    transformed_data = []

    original_index = df.index

    if target_column in df.columns:
        target = df[target_column].copy()
    else:
        target = None
    if weight_column in df.columns:
        weight = df[weight_column].copy()
    else:
        weight = None

    numerical_data = df[numerical_features].copy()
    categorical_data = df[categorical_features].copy()

    # log
    numerical_transformed = numerical_data.apply(lambda x: np.log(x + 1))

    # scale
    numerical_scaled = preprocess_numerical_data(df=numerical_transformed,
                                                 numerical_features=numerical_features)

    # pca
    pca = PCA()
    numerical_pca = pd.DataFrame(
        pca.fit_transform(numerical_scaled),
        columns=numerical_features,
        index=original_index)

    transformed_data.append(numerical_pca)

    if categorical_data is not None:
        transformed_data.append(categorical_data)

    if weight is not None:
        transformed_data.append(weight)

    transformed_df = pd.concat(transformed_data, axis=1)

    if target is not None:
        transformed_df[target_column] = target
        transformed_df[target_column] = transformed_df[target_column].astype(int)

    transformed_df.index = original_index

    return transformed_df

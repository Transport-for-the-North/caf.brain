# -*- coding: utf-8 -*-
"""
Created on: 1/17/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import linear_rainbow
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from caf.ml.process_data_functions.encode_and_scale import preprocess_numerical_data
from sklearn.linear_model import (Ridge,
                                  Lasso,
                                  ElasticNet,
                                  LinearRegression)

def pre_forecast_data_analysis(residuals,
                               model,
                               x_test,
                               train_scaled,
                               test_scaled,
                               full_transformations,
                               train_unscaled,
                               test_unscaled,
                               numerical_features,
                               categorical_features,
                               target_column,
                               weight_column,
                               x_train,
                               y_train):

    alpha = 0.05
    train_final = None
    test_final = None
    linearity_present = False
    normality_present = False
    multicolinearity_present = False
    autocorrelation_present = False
    heteroscedasticity_present = False

    if isinstance(model, (LinearRegression, Lasso, Ridge, ElasticNet)):
        # Statsmodels conversion and tests
        X_with_const_train = add_constant(x_train)
        ols_model = OLS(y_train, X_with_const_train).fit()
        residuals = ols_model.resid

        # Rainbow Test for Linearity
        rainbow_statistic, rainbow_p_value = linear_rainbow(ols_model)
        print(f"Rainbow test p-value: {rainbow_p_value}")
        if rainbow_p_value < alpha:
            print("Warning: Rainbow test suggests non-linearity.")
            linearity_present = True

        # Shapiro-Wilk Test for Normality
        shapiro_statistic, shapiro_p_value = shapiro(residuals)
        print(f"Shapiro-Wilk test p-value: {shapiro_p_value}")
        if shapiro_p_value < alpha:
            print("Warning: Shapiro-Wilk test suggests non-normality of residuals.")
            normality_present = True

        # multicolinearity
        print('Checking for multicollinearity')
        vif_data = pd.DataFrame()
        vif_data["Feature"] = train_scaled.columns
        vif_data["VIF"] = [variance_inflation_factor(train_scaled.values, i) for i in range(train_scaled.shape[1])]
        high_vif = vif_data[vif_data["VIF"] > 10]
        if not high_vif.empty:
            print("High VIF variables:")
            print(high_vif)
            multicolinearity_present = True
    else:
        # Autocorrelation
        dw_statistic = durbin_watson(residuals)
        print(f"Durbin-Watson statistic: {dw_statistic}")
        if dw_statistic < 1.5 or dw_statistic > 2.5:
            print("Warning: Potential autocorrelation in residuals.")
            autocorrelation_present = True

        # Breusch-Pagan Test for Heteroscedasticity
        X_with_const = add_constant(x_test)
        print('Checking for heteroscedasticity')
        bp_test_statistic, bp_test_p_value, _, _ = het_breuschpagan(residuals, X_with_const)
        print(f"Breusch-Pagan test p-value: {bp_test_p_value}")
        if bp_test_p_value < alpha:
            print("Warning: Breusch-Pagan test suggests heteroscedasticity.")
            heteroscedasticity_present = True

        # White's Test for Heteroscedasticity
        white_test_statistic, white_test_p_value, _, _ = het_white(residuals, X_with_const)
        print(f"White's test p-value: {white_test_p_value}")
        if white_test_p_value < alpha:
            print("Warning: White's test suggests heteroscedasticity.")
            heteroscedasticity_present = True

    if (linearity_present or
        normality_present or
        multicolinearity_present or
        autocorrelation_present or
        heteroscedasticity_present) and full_transformations:
        print('Data issue present, corrective transformations applied to numerical features')
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
            return train_final, test_final
        else:
            return train_scaled, test_scaled

    elif (linearity_present or normality_present or
          multicolinearity_present or autocorrelation_present or
          heteroscedasticity_present) and full_transformations is False:

            print('Data issue present but transformations are not permitted by the user.')
            return train_scaled, test_scaled
    else:
        print('No data issues present.')
        return train_scaled, test_scaled


def transform_data(df,
                   numerical_features,
                   categorical_features,
                   target_column,
                   weight_column):
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

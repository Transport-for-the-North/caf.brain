# -*- coding: utf-8 -*-
"""
Created on: 4/2/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import boxcox
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor,
                              BaggingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.diagnostic import linear_rainbow, het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

from caf.ml.functions.process_data_functions import convert_to_dataframe
from caf.ml.inputs.cafml_inputs import Models


def pre_forecast_data_analysis(data,
                               regression_method,
                               target_column,
                               threshold,
                               threshold_corr,
                               output_folder):
    alpha = 0.05
    x = data.drop(columns=[target_column])
    y = data[target_column]
    any_issue_present = False
    dataframe = None

    if regression_method in [Models.RIDGE, Models.LASSO, Models.ELASTICNET,
                             Models.LINEAR_REGRESSION, Models.SVR]:
        print('-------------------------------------------------')
        print('Assessing linearity, normality and heteroscedasticity')
        # Convert scikit-learn model to statsmodels OLS equivalent
        X_with_const = add_constant(x)
        ols_model = OLS(y, X_with_const).fit()

        # Use statsmodels model for assumption checks
        residuals = ols_model.resid
        exog = ols_model.model.exog

        # Rainbow Test for Linearity
        rainbow_statistic, rainbow_p_value = linear_rainbow(ols_model)
        print(f"Rainbow test p-value: {rainbow_p_value}")
        if rainbow_p_value < alpha:
            print("Warning: Rainbow test suggests non-linearity.")
            any_issue_present = True

        # Shapiro-Wilk Test for Normality
        shapiro_statistic, shapiro_p_value = shapiro(residuals)
        print(f"Shapiro-Wilk test p-value: {shapiro_p_value}")
        if shapiro_p_value < alpha:
            print("Warning: Shapiro-Wilk test suggests non-normality of residuals.")
            any_issue_present = True

        # Breusch-Pagan Test for Heteroscedasticity
        bp_test_statistic, bp_test_p_value, _, _ = het_breuschpagan(residuals, exog)
        print(f"Breusch-Pagan test p-value: {bp_test_p_value}")
        if bp_test_p_value < alpha:
            print("Warning: Breusch-Pagan test suggests heteroscedasticity.")
            any_issue_present = True

        # White's Test for Heteroscedasticity
        white_test_statistic, white_test_p_value, _, _ = het_white(residuals, exog)
        print(f"White's test p-value: {white_test_p_value}")
        if white_test_p_value < alpha:
            print("Warning: White's test suggests heteroscedasticity.")
            any_issue_present = True

    if any_issue_present:
        print('------------------------------------------------------------')
        print('Due to warnings, Logarithmic transformation being applied')

        transformed_data = data.applymap(lambda x: np.log(x + 1))
        dataframe = assess_multicolinearity(dataframe=transformed_data,
                                            input_data=data,
                                            target_column=target_column,
                                            threshold=threshold,
                                            threshold_corr=threshold_corr)

        scale = StandardScaler()
        x_scaled = scale.fit_transform(dataframe)

        # Convert to DataFrame if needed
        dataframe = convert_to_dataframe(x_scaled)
        dataframe = pd.DataFrame(data=x_scaled, columns=dataframe.columns.values,
                                index=dataframe.index.values)
        dataframe = dataframe.astype(float)

        # Add y back to the dataframe
        dataframe[target_column] = y.values

        output_filename = 'Final_data_to_model.csv'
        output_path = os.path.join(output_folder, output_filename)
        dataframe.to_csv(output_path, index=True)
        print(f"Final data to model exported to: {output_path}")

        return dataframe


    if regression_method in [Models.RANDOM_FOREST]:
        # need to consider: Nonlinearity, Overfitting, Model Complexity
        pass
    if regression_method in [Models.EXTRA_TREES]:
        # need to consider: Nonlinearity, Overfitting, Model Complexity
        pass
    if regression_method in [Models.GRADIENT_BOOSTING]:
        # need to consider: Nonlinearity, Overfitting, Model Complexity
        pass
    if regression_method in [Models.ADABOOST]:
        # need to consider: Nonlinearity, Overfitting, Model Complexity
        pass
    if regression_method in [Models.BAGGING]:
        # need to consider: Nonlinearity, Overfitting, Model Complexity
        pass
    if regression_method in [Models.KNN]:
        pass
    if regression_method in [Models.DECISION_TREE]:
        # need to consider: Nonlinearity, Overfitting
        pass
    if regression_method in [Models.NEURAL_NETWORK]:
        # need to consider: Nonlinearity, Model Complexity
        pass

    dataframe = assess_multicolinearity(dataframe=data,
                                        input_data=None,
                                        target_column=target_column,
                                        threshold=threshold,
                                        threshold_corr=threshold_corr)

    scale = StandardScaler()
    x_scaled = scale.fit_transform(dataframe)

    # Convert to DataFrame if needed
    dataframe = convert_to_dataframe(x_scaled)
    dataframe = pd.DataFrame(data=x_scaled, columns=dataframe.columns.values,
                             index=dataframe.index.values)
    dataframe = dataframe.astype(float)

    # Add y back to the dataframe
    dataframe[target_column] = y.values

    output_filename = 'Final_data_ready_to_model.csv'
    output_path = os.path.join(output_folder, output_filename)
    dataframe.to_csv(output_path, index=True)
    print(f"Final data ready to model exported to: {output_path}")

    return dataframe


def assess_multicolinearity(dataframe,
                            input_data,
                            target_column,
                            threshold,
                            threshold_corr):
    print('----------------------------------------------------------------')
    print('Multicollinearity analysis underway')
    #### VARIANCE INFLATION FACTOR ####
    X = None
    y = None
    # No log transformation
    if dataframe is not None and not dataframe.empty:
        X = dataframe
        y = dataframe[target_column]

    # Post log transformation
    if input_data is not None and not input_data.empty:
        X = dataframe
        y = input_data[target_column]

    if X is not None:
        print('VIF analysis underway')
        # Check for multicollinearity using VIF
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

        # Identify variables with high VIF
        high_vif_variables = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()

        if len(high_vif_variables) == 0:
            print('All columns are highly correlated based on VIF. Please reassess data')
            return dataframe

        #### CORRELATION MATRIX ####
        print('Correlation Matrix analysis underway')
        # Calculate the correlation matrix for numeric columns
        correlation_matrix = pd.DataFrame(X_scaled, columns=X.columns).corr()

        # Identify highly correlated columns
        high_correlation_columns = correlation_matrix.columns[
            (correlation_matrix.abs() > threshold_corr).any(axis=0)].tolist()

        if len(high_correlation_columns) == 0:
            print('All columns are highly correlated based on correlation matrix. Please reassess data')
            return dataframe

        #### OUTCOMES ####
        print('High VIF variables:')
        print(high_vif_variables)
        print(len(high_vif_variables))

        print('High correlation matrix variables:')
        print(high_correlation_columns)
        print(len(high_correlation_columns))

        #### PCA FIX ####
        if len(high_vif_variables) >= 0.2 * X.shape[1] or len(high_correlation_columns) >= 0.2 * X.shape[1]:
            print('Fixing multicolinearity issues through PCA transformation')
            # Apply PCA
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            X_pca = pd.DataFrame(X_pca, columns=X.columns, index=y.index)

            # merge the PCA data
            dataframe_with_selected_features_pca = pd.merge(X_pca, y, left_index=True, right_index=True, how='outer')

            return dataframe_with_selected_features_pca

    return dataframe

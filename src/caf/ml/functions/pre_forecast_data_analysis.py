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
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor,
                              BaggingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.diagnostic import linear_rainbow, het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant

from caf.ml.functions.data_pipeline_functions import preprocess_categorical_data, \
    preprocess_numerical_data, process_data_pipeline
from caf.ml.functions.feature_selection import identify_feature_types
from caf.ml.functions.process_data_functions import convert_to_dataframe
from caf.ml.inputs.cafml_inputs import Models


def pre_forecast_data_analysis(data,
                               regression_method,
                               target_column,
                               threshold,
                               threshold_corr,
                               output_folder,
                               index_col,
                               categorical_data,
                               categorical_features,
                               categorical_transformations):
    alpha = 0.05
    x_ = data.drop(columns=[target_column])
    y = data[target_column]
    any_issue_present = False
    dataframe = None
    transformations = []

    if categorical_transformations is not None:
        transformations.extend(categorical_transformations)

    if regression_method in [Models.LOGIT_REGRESSION_L1, Models.LOGIT_REGRESSION_L2,
                             Models.LOGIT_REGRESSION_ELASTICNET, Models.MULTINOMIAL, Models.PROBIT]:
        print('-------------------------------------------------')
        print('Assessing assumptions for logistic/multinomial/probit regression')

        x = pd.get_dummies(x_, columns=categorical_features, drop_first=True)
        model_fit = regression_method.fit()
        X_with_const = add_constant(x)

        # Use statsmodels model for assumption checks
        if regression_method == Models.PROBIT:
            residuals = model_fit.resid
            exog = model_fit.model.exog
        else:
            residuals = y - regression_method.predict(X_with_const)
            exog = X_with_const

        # Linearity
        print('Checking linearity')
        for col in x.columns:
            logit = np.log((y.value_counts()[1] + 1e-5) / (y.value_counts()[0] + 1e-5))
            linearity_check = pd.DataFrame({
                'x': x[col],
                'logit': logit
            })
            linearity_check = linearity_check.groupby('x').mean()
            if not linearity_check.corr().iloc[0, 1] > 0.9:
                print(f"Warning: {col} may not be linearly related to the logit.")
                any_issue_present = True

        # Multicollinearity
        print('Checking for multicollinearity')
        vif_data = pd.DataFrame()
        vif_data["Feature"] = x.columns
        vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        high_vif = vif_data[vif_data["VIF"] > threshold]
        if not high_vif.empty:
            print("High VIF variables:")
            print(high_vif)
            any_issue_present = True

        # Autocorrelation
        dw_statistic = durbin_watson(residuals)
        print(f"Durbin-Watson statistic: {dw_statistic}")
        if dw_statistic < 1.5 or dw_statistic > 2.5:
            print("Warning: Potential autocorrelation in residuals.")
            any_issue_present = True

        # Heteroscedasticity
        print('Checking for heteroscedasticity')
        bp_test_statistic, bp_test_p_value, _, _ = het_breuschpagan(residuals, exog)
        print(f"Breusch-Pagan test p-value: {bp_test_p_value}")
        if bp_test_p_value < alpha:
            print("Warning: Breusch-Pagan test suggests heteroscedasticity.")
            any_issue_present = True

    if regression_method in [Models.RIDGE, Models.LASSO, Models.ELASTICNET,
                             Models.LINEAR_REGRESSION, Models.SVR]:
        print('-------------------------------------------------')
        print('Assessing linearity, normality and heteroscedasticity')
        # Convert scikit-learn model to statsmodels OLS equivalent
        X_with_const = add_constant(x_)
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


    if regression_method in [Models.RANDOM_FOREST, Models.EXTRA_TREES,
                             Models.GRADIENT_BOOSTING, Models.ADABOOST,
                             Models.BAGGING, Models.DECISION_TREE]:
        pass

    if regression_method in [Models.KNN]:
        pass

    if regression_method in [Models.NEURAL_NETWORK]:
        pass

    if any_issue_present and categorical_data is not None:
        transformed_data = x_.applymap(lambda x: np.log(x + 1))
        transformations.append(('log', None))

        pca = PCA()
        dataframe = pca.fit_transform(transformed_data)
        transformations.append(('PCA', pca))


        dataframe = pd.DataFrame(dataframe, columns=x_.columns, index=data.index)

        dataframe_final = pd.concat([dataframe, y], axis=1)

        if output_folder is not None:
            output_filename = 'Final_data_ready_to_model.csv'
            output_path = os.path.join(output_folder, output_filename)
            dataframe_final.to_csv(output_path, index=True)
            print(f"Final data to model exported to: {output_path}")

        print("Final_data_ready_to_model:")
        print(dataframe_final.shape)
        print(dataframe_final)

        return dataframe_final, transformations


    elif not any_issue_present and categorical_data is not None:

        if output_folder is not None:
            output_filename = 'Final_data_ready_to_model.csv'
            output_path = os.path.join(output_folder, output_filename)
            data.to_csv(output_path, index=True)
            print(f"Final data to model exported to: {output_path}")

        print("Final_data_ready_to_model:")
        print(data.shape)
        print(data)

        return data, transformations


    elif any_issue_present:
        print('------------------------------------------------------------')
        print('Due to warnings, Logarithmic transformation being applied')

        transformed_data = x_.applymap(lambda x: np.log(x + 1))
        transformations.append(('log', None))
        dataframe, transformations_done = assess_multicolinearity(dataframe=transformed_data,
                                            input_data=data,
                                            target_column=target_column,
                                            threshold=threshold,
                                            threshold_corr=threshold_corr)
        transformations.extend(transformations_done)

        if not isinstance(dataframe, pd.DataFrame):
            dataframe = pd.DataFrame(dataframe, columns=dataframe.columns, index=data.index)

        dataframe = dataframe.astype(float)

        if output_folder is not None:
            output_filename = 'Final_data_ready_to_model.csv'
            output_path = os.path.join(output_folder, output_filename)
            dataframe.to_csv(output_path, index=True)
            print(f"Final data to model exported to: {output_path}")

        print("Final_data_ready_to_model:")
        print(dataframe.shape)
        print(dataframe)

        return dataframe, transformations

    dataframe, transformations_done = assess_multicolinearity(dataframe=data,
                                        input_data=None,
                                        target_column=target_column,
                                        threshold=threshold,
                                        threshold_corr=threshold_corr)
    transformations.extend(transformations_done)

    if not isinstance(dataframe, pd.DataFrame):
        dataframe = pd.DataFrame(dataframe, columns=dataframe.columns, index=data.index)

    dataframe = dataframe.astype(float)

    if output_folder is not None:
        output_filename = 'Final_data_ready_to_model.csv'
        output_path = os.path.join(output_folder, output_filename)
        dataframe.to_csv(output_path, index=True)
        print(f"Final data ready to model exported to: {output_path}")

    print("Final_data_ready_to_model:")
    print(dataframe.shape)
    print(dataframe)

    return dataframe, transformations


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
    transformations_done = []
    # No log transformation
    if dataframe is not None and not dataframe.empty:
        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]

    # Post log transformation
    if input_data is not None and not input_data.empty:
        X = dataframe.drop(columns=[target_column])
        y = input_data[target_column]

    if X is not None:
        print('VIF analysis underway')
        # Check for multicollinearity using VIF
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        transformations_done.append(('scaling', scaler))
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

        # Identify variables with high VIF
        high_vif_variables = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()

        if len(high_vif_variables) == 0:
            print('All columns are highly correlated based on VIF. Please reassess data. Data has been scaled')
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=y.index)
            dat = pd.merge(X_scaled, y, left_index=True, right_index=True, how='outer')
            return dat, transformations_done

        #### CORRELATION MATRIX ####
        print('Correlation Matrix analysis underway')
        # Calculate the correlation matrix for numeric columns
        correlation_matrix = pd.DataFrame(X_scaled, columns=X.columns).corr()

        # Identify highly correlated columns
        high_correlation_columns = correlation_matrix.columns[
            (correlation_matrix.abs() > threshold_corr).any(axis=0)].tolist()

        if len(high_correlation_columns) == 0:
            print('All columns are highly correlated based on correlation matrix. Please reassess data. Data has been scaled.')
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=y.index)
            dat = pd.merge(X_scaled, y, left_index=True, right_index=True, how='outer')
            return dat, transformations_done

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
            transformations_done.append(('PCA', pca))
            X_pca = pd.DataFrame(X_pca, columns=X.columns, index=dataframe.index)

            dataframe_with_selected_features_pca = pd.concat([X_pca, y], axis=1)

            return dataframe_with_selected_features_pca, transformations_done

    if input_data is not None and not input_data.empty:
        if target_column not in dataframe.columns:
            dataframe[target_column] = y.values

    return dataframe, transformations_done


def apply_transformations(predict_data, transformations, target_column, numerical_features, categorical_features, output_folder):

    transformed_data = predict_data.copy()
    columns_changed = False
    new_columns = predict_data.columns.values

    for transform_name, transform_obj in transformations:
        if transform_name == 'Scaling and encoding':
            transformed_data, _ = process_data_pipeline(df=transformed_data,
                                                        numerical_features=numerical_features,
                                                        categorical_features=categorical_features,
                                                        target_column=target_column,
                                                        output_folder=output_folder)
            if isinstance(transformed_data, tuple):
                transformed_data = transformed_data[0]
            columns_changed = True
            new_columns = transformed_data.columns.values
        elif transform_name == 'log':
            transformed_data = transformed_data.apply(lambda x: np.log(x + 1))

        elif transform_name == 'scaling':
            scaler = transform_obj
            transformed_data = scaler.transform(transformed_data)

        elif transform_name == 'PCA':
            pca = transform_obj
            transformed_data = pca.transform(transformed_data)


        print(f"Transformation: {transform_name}")
        print(f"Type of transformed_data: {type(transformed_data)}")
        if isinstance(transformed_data, (pd.DataFrame, np.ndarray)):
            print(f"Shape of transformed_data: {transformed_data.shape}")

    data = convert_to_dataframe(transformed_data, columns=new_columns, index=predict_data.index)

    if data is None:
        raise ValueError('Transformation application failed')

    final_predict_data = data.astype(float)

    print("Predict_data_post_transformations:")
    print(final_predict_data.shape)
    print(final_predict_data)

    return final_predict_data

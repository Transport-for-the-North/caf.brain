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

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from caf.ml.backlog.functions_to_be_processed import  process_data_pipeline
from caf.ml.backlog.functions_to_be_processed import convert_to_dataframe

'''def pre_forecast_data_analysis(data,
                               regression_method,
                               target_column,
                               threshold,
                               threshold_corr,
                               output_folder,
                               index_col,
                               categorical_data,
                               categorical_features,
                               categorical_transformations,
                               features_to_transform):
    print(regression_method)
    alpha = 0.05
    x_ = data.drop(columns=[target_column])
    y = data[target_column]
    any_issue_present = False
    dataframe = None
    transformations = []

    if categorical_transformations is not None:
        transformations.extend(categorical_transformations)
    if isinstance(regression_method, (LogisticRegression, ElasticNet)):
        print('-------------------------------------------------')
        print('Assessing assumptions for logistic/multinomial/probit regression')

        # Fit the model
        model_fit = regression_method.fit(x_, y)
        y_pred = model_fit.prediction_model(x_)
        y = pd.to_numeric(y, errors='coerce')
        y_pred = pd.to_numeric(y_pred, errors='coerce')
        residuals = y - y_pred

        if isinstance(regression_method, LogisticRegression):
            y_pred_proba = model_fit.predict_proba(x_)[:, 1]

        # Linearity
        print('Checking linearity')
        for col in x_.columns:
            correlation = np.corrcoef(x_[col], residuals)[0, 1]
            if abs(correlation) > 0.1:
                print(f"Warning: {col} may not be linearly related to the target.")
                any_issue_present = True

        # multicolinearity
        print('Checking for multicollinearity')
        vif_data = pd.DataFrame()
        vif_data["Feature"] = x_.columns
        vif_data["VIF"] = [variance_inflation_factor(x_.values, i) for i in range(x_.shape[1])]
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
        X_with_const = add_constant(x_)
        bp_test_statistic, bp_test_p_value, _, _ = het_breuschpagan(residuals, X_with_const)
        print(f"Breusch-Pagan test p-value: {bp_test_p_value}")
        if bp_test_p_value < alpha:
            print("Warning: Breusch-Pagan test suggests heteroscedasticity.")
            any_issue_present = True

        print(f'Any issue present: {any_issue_present}')
    if isinstance(regression_method, (Ridge, Lasso, ElasticNet, LinearRegression, SVR)):

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


    if any_issue_present and categorical_data is not None:
        transformed_data = x_.map(lambda x: np.log(x + 1))
        transformations.append(('log', None))

        ##experimental functions_to_be_processed##
        df_final_to_model, transformations = experimental_functions(data=transformed_data,
                                                                    categorical_transformations=transformations,
                                                                    features_to_interact=x_.columns,
                                                                    features_to_transform=features_to_transform,
                                                                    output_folder=output_folder)


        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        numerical_data = numerical_pipeline.fit_transform(df_final_to_model)
        transformations.append(('scaling', None))

        pca = PCA()
        dataframe = pca.fit_transform(numerical_data)
        transformations.append(('PCA', pca))


        # dataframe = pd.DataFrame(dataframe, index=data.index)
        dataframe = pd.DataFrame(dataframe, columns=df_final_to_model.columns, index=data.index)

        dataframe_final = pd.concat([dataframe, y], axis=1)
        print(dataframe_final)
        print(dataframe_final.shape)
        print(dataframe_final.columns)
        if output_folder is not None:
            output_filename = 'data_analysis_dataframe.csv'
            output_path = os.path.join(output_folder, output_filename)
            dataframe_final.to_csv(output_path, index=True)
            print(f"data_analysis_dataframe: {output_path}")

        print("data_analysis_dataframe:")
        print(dataframe_final.shape)
        print(dataframe_final)
        print(dataframe_final.columns)

        return dataframe_final, transformations


    elif not any_issue_present and categorical_data is not None:

        if output_folder is not None:
            output_filename = 'data_analysis_dataframe.csv'
            output_path = os.path.join(output_folder, output_filename)
            data.to_csv(output_path, index=True)
            print(f"data_analysis_dataframe: {output_path}")

        print("data_analysis_dataframe:")
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
            output_filename = 'data_analysis_dataframe.csv'
            output_path = os.path.join(output_folder, output_filename)
            dataframe.to_csv(output_path, index=True)
            print(f"data_analysis_dataframe: {output_path}")

        print("data_analysis_dataframe:")
        print(dataframe.shape)
        print(dataframe)
        print(dataframe.columns)
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
        output_filename = 'data_analysis_dataframe.csv'
        output_path = os.path.join(output_folder, output_filename)
        dataframe.to_csv(output_path, index=True)
        print(f"data_analysis_dataframe: {output_path}")

    print("data_analysis_dataframe:")
    print(dataframe.shape)
    print(dataframe)
    print(dataframe.columns)

    return dataframe, transformations
'''
"""def assess_multicolinearity(dataframe,
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

    return dataframe, transformations_done"""


def apply_transformations(predict_data, transformations,
                          target_column, numerical_features,
                          categorical_features, output_folder,
                          training_data, features_to_transform):

    transformed_data = predict_data.copy()
    columns_changed = False
    new_columns = predict_data.columns.values
    # transformed_data['car'] = 0

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
            print(transformed_data.columns)
        elif transform_name == 'log':
            transformed_data = transformed_data.apply(lambda x: np.log(x + 1))


        elif transform_name == 'interaction_terms_and_poly_features':
            transformed_data, _ = experimental_functions(data=transformed_data,
                                                         categorical_transformations=transformations,
                                                         features_to_interact=transformed_data.columns,
                                                         features_to_transform=features_to_transform,
                                                         output_folder=output_folder)

        elif transform_name == 'scaling':
            # scaler = transform_obj
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            scaled_data = numerical_pipeline.fit_transform(transformed_data)
            transformed_data = pd.DataFrame(scaled_data, columns=transformed_data.columns, index=transformed_data.index)

            # transformed_data = scaler.transform(transformed_data)

        elif transform_name == 'PCA':
            print(training_data.columns)
            training_data = training_data.drop(columns=target_column)
            transformed_data = transformed_data[training_data.columns]
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
    print(final_predict_data)
    return final_predict_data



def experimental_functions(data, categorical_transformations, features_to_interact, features_to_transform, output_folder):
    transformations = []
    transformations.extend(categorical_transformations)

    if isinstance(features_to_interact, pd.Index):
        features_to_interact = list(features_to_interact)

    interaction_terms = {}
    for i, f1 in enumerate(features_to_interact):
        for f2 in features_to_interact[i + 1:]:
            interaction_terms[f'{f1}_{f2}_interaction'] = data[f1] * data[f2]

    interaction_df = pd.DataFrame(interaction_terms)
    final_df = pd.concat([data, interaction_df], axis=1)


    if features_to_transform is None or len(features_to_transform) == 0:
        features_to_transform = data.columns.tolist()

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    if len(features_to_transform) > 10:  # Arbitrary threshold, adjust as needed
        features_to_transform = features_to_transform[:10]
    poly_features = poly.fit_transform(final_df[features_to_transform])
    # feature_names = poly.get_feature_names_out(features_to_transform)

    feature_names = []
    for feature_indices, _ in zip(poly.powers_, poly_features.T):
        feature_name = ' * '.join(
            [features_to_transform[i] for i, p in enumerate(feature_indices) if p > 0])
        feature_names.append(feature_name)

    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=final_df.index)

    for col in poly_df.columns:
        if col in final_df.columns:
            poly_df = poly_df.rename(columns={col: f'poly_{col}'})

    final_df = pd.concat([final_df, poly_df], axis=1)

    transformations.append(('interaction_terms_and_poly_features', None))


    output_filename = 'experimental_function_results.csv'
    output_path = os.path.join(output_folder, output_filename)
    final_df.to_csv(output_path, index=True)
    print(f'Experimental function results: {final_df.shape}')

    return final_df, transformations

# -*- coding: utf-8 -*-
"""
Created on: 10/3/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
import numpy as np
import pandas as pd
from caf.ml.functions.data_pipeline_functions import process_data_pipeline
from caf.ml.functions.feature_selection import get_cv_class
from caf.ml.functions.forecast_model_functions import process_forecast_data, align_dataframes, predict_refined
from caf.ml.functions.model_algorithm_evaluation import eval_model
from caf.ml.functions.pre_forecast_data_analysis import pre_forecast_data_analysis, experimental_functions
from caf.ml.functions.process_data_functions import convert_to_dataframe
from caf.ml.functions.save_model import save_model_and_parameters
from caf.ml.inputs.cafml_inputs import Models, Default_regression_methods, Models_List_
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


def norcom_run_functions(params, preprocessed_df, transformations_):
    valid_models = [model.value for model in Models] + Default_regression_methods + Models_List_
    model_type = params.model_type
    if model_type is None:
        model_type = Default_regression_methods
    elif not isinstance(model_type, list):
        model_type = [model_type]
    for mod in model_type:
        if mod is not None and mod not in valid_models:
            raise ValueError(
                f"Invalid model type: {mod}. Please provide valid model types from Models class or Default_regression_methods.")

    model = model_type[0].get_model_instance() if isinstance(model_type[0], Models) else model_type[0]
    print(f'Model to be used: {model}')

    df, transformations = pre_forecast_data_analysis(data=preprocessed_df,
                                                     regression_method=model,
                                                     target_column=params.target_column,
                                                     threshold=None,
                                                     threshold_corr=None,
                                                     output_folder=params.output_folder,
                                                     index_col=params.index_columns,
                                                     categorical_data=params.categorical_data,
                                                     categorical_features=params.categorical_features,
                                                     categorical_transformations=transformations_,
                                                     features_to_transform=None)

    data = quick_feature_selection(data=df, target_column=params.target_column, cv=5, regression_method=model)

    output_path = os.path.join(params.output_folder, 'data_used_to_train_model_final.csv')
    data.to_csv(output_path, index=True)

    hyperparameters = modified_hyper_optimisation(regression_method=model,
                                                  data=data,
                                                  target_column=params.target_column,
                                                  output_folder=params.output_folder)


    save_model_and_parameters(regression_method=model,
                              hyperparameters=hyperparameters,
                              transformations=transformations,
                              output_folder=params.output_folder,
                              skip_data_analysis=None,
                              skip_hyperparameter_optimisation=None)


    ## FINAL PREDICTION ##
    predict_data = process_forecast_data(df=params.predict_data,
                                         index_columns_p=params.index_columns_predict,
                                         drop_columns_p=params.drop_columns_predict,
                                         target_column=params.target_column,
                                         keep_columns_p=params.keep_columns_predict,
                                         outlier_threshold_p=None,
                                         categorical_target=params.categorical_target,
                                         output_folder=params.output_folder)

    dat = apply_transformations_norcom(predict_data=predict_data,
                                       transformations=transformations,
                                       target_column=params.target_column,
                                       numerical_features=params.numerical_features,
                                       categorical_features=params.categorical_features,
                                       output_folder=params.output_folder,
                                       training_data=data,
                                       features_to_transform=None,
                                       training_data_pre_feat_selection=df)

    final_predict_data = align_dataframes(df1=data, df2=dat,
                                          output_folder=params.output_folder)

    forecasted_data, y_proba = predict_refined(trained_data=data,
                                               predict_data=final_predict_data,
                                               trained_model=model,
                                               target_column=params.target_column,
                                               output_folder=params.output_folder,
                                               FinalModelParameters=hyperparameters,
                                               index_col=params.index_columns)

    print(forecasted_data)

    eval_model(data_used_to_predict=final_predict_data,
               data_contains_truth_values_only=params.validation_data,
               model_predicted_data=forecasted_data,
               model=model,
               target_column=params.target_column,
               output_folder=params.output_folder,
               categorical_target=params.categorical_target,
               y_proba=y_proba)

    return


def norcom_feature_selection(data, model, cv_method, splits, repeats, target_column,  multiple_year_prediction, categorical_data):
    print('test')
    if isinstance(model, LogisticRegression):
        model.set_params(max_iter=1000)


    x = data.drop(columns=[target_column])
    y = data[target_column]


    best_features_model = []
    best_features_f_regressor = []
    best_features_rfe = []
    best_score_f_regressor = float('-inf')
    best_score_rfe = float('-inf')

    best_features_sfs = []
    best_features_mutual_info = []


    cv = TimeSeriesSplit(n_splits=5) if multiple_year_prediction is not None else get_cv_class(
        cv_method, splits=splits, repeats=repeats)

    if categorical_data is not None:
        for train_index, test_index in tqdm(cv.split(x, y), total=cv.get_n_splits(x, y),
                                            desc="Feature selection progress"):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)

            # RFECV for feature selection
            rfecv = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy')
            selector = rfecv.fit(X_train, y_train)
            X_train_rfe = selector.transform(X_train)
            score_rfe = cross_val_score(model, X_train_rfe, y_train, scoring='accuracy',
                                        cv=cv).mean()
            if score_rfe > best_score_rfe:
                best_score_rfe = score_rfe
                support_mask_rfe = selector.support_
                best_features_rfe = X_train.columns[support_mask_rfe].to_list()

            # Mutual Information for feature selection
            selector_mi = SelectKBest(mutual_info_classif, k='all').fit(X_train, y_train)
            X_selected_mi = selector_mi.transform(X_train)
            score_f_regressor = cross_val_score(model, X_selected_mi, y_train, scoring='accuracy',
                                                cv=cv).mean()
            if score_f_regressor > best_score_f_regressor:
                best_score_f_regressor = score_f_regressor
                support_mask_mi = selector_mi.get_support()
                best_features_f_regressor = X_train.columns[support_mask_mi].to_list()


        if best_score_f_regressor != float('-inf'):
            print("Best score for mutual_info_classif:", best_score_f_regressor)
        if best_score_rfe != float('-inf'):
            print("Best score for RFE:", best_score_rfe)
        print('Trained model', model)

        return best_features_model, best_features_f_regressor, best_features_rfe, best_features_sfs, best_features_mutual_info


def quick_feature_selection(data, target_column, cv, regression_method):
    print(regression_method)

    if isinstance(regression_method, LogisticRegression):
        regression_method.set_params(max_iter=1000)

    print('Feature selection beginning')
    X = data.drop(columns=[target_column])
    y = data[target_column]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # tqdm progress bar
    with tqdm(total=1, desc="Fitting RandomForest") as pbar:
        rf.fit(X, y)
        pbar.update(1)

    # SelectFromModel to select features
    selector = SelectFromModel(rf, prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()


    # tqdm progress bar for cv
    with tqdm(total=cv, desc="Cross-validation") as pbar:
        final_score = cross_val_score(regression_method, X[selected_features], y, cv=cv, scoring='roc_auc',
                                      n_jobs=-1, verbose=0)
        pbar.update(cv)

    print(f"Number of features selected: {len(selected_features)}")
    print(f"Cross-validated ROC AUC score: {final_score.mean()}")

    dataframe_final = pd.concat([X[selected_features], y], axis=1)
    print(dataframe_final)
    print(dataframe_final.shape)
    return dataframe_final


def modified_hyper_optimisation(regression_method, data, target_column, output_folder):
    print('Hyperparameter optimisation underway')
    x = data.drop(columns=[target_column])
    y = data[target_column]
    cv = TimeSeriesSplit(n_splits=5)

    def _grid_search(model_instance,
                     param_grid,
                     cv,
                     scoring):
        grid_search = GridSearchCV(model_instance, param_grid, cv=cv, scoring=scoring, verbose=2)
        grid_search.fit(x, y)
        best_params = grid_search.best_params_
        print('Best parameters for model are:')
        print(best_params)
        print('CV results:')
        print(grid_search.cv_results_)
        return best_params

    param_grid = None
    if isinstance(regression_method, LogisticRegression):
        if regression_method.penalty == 'elasticnet':
            print('Elasticnet parameter grid selected')
            param_grid = {"C": [1.0, 0.1, 0.01, 0.001], "l1_ratio": [0.1, 0.5, 0.9]}
        elif regression_method.penalty == 'l1':
            print('L1 parameter grid selected')
            param_grid = {"C": [1.0, 0.1, 0.01, 0.001]}
        elif regression_method.penalty == 'l2':
            print('L2 parameter grid selected')
            param_grid = {"C": [1.0, 0.1, 0.01, 0.001]}


    best_params = _grid_search(regression_method, param_grid, cv=cv, scoring="accuracy")
    best_model = regression_method.set_params(**best_params)
    best_model.fit(x, y)

    # Extract coefficients
    feature_names = x.columns.tolist()
    print(feature_names)
    coefficients = best_model.coef_[0]
    print(coefficients)
    intercept = best_model.intercept_[0]
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['Intercept'] = intercept

    coef_df.to_csv(os.path.join(output_folder, 'coefficients_from_trained_model.csv'), index=True)

    return best_params


def apply_transformations_norcom(predict_data, transformations,
                                 target_column, numerical_features,
                                 categorical_features, output_folder,
                                 training_data, features_to_transform,
                                 training_data_pre_feat_selection):

    transformed_data = predict_data.copy()
    columns_changed = False
    new_columns = predict_data.columns.values
    # transformed_data['car'] = 0

    for transform_name, transform_obj in transformations:
        if transform_name == 'Scaling and encoding':
            """transformed_data, _ = process_data_pipeline(df=transformed_data,
                                                        numerical_features=numerical_features,
                                                        categorical_features=categorical_features,
                                                        target_column=target_column,
                                                        output_folder=output_folder)"""

            transformed_data, _ = alternative_encoding(data=transformed_data,
                                                       target_column=target_column,
                                                       output_folder=output_folder)

            if isinstance(transformed_data, tuple):
                transformed_data = transformed_data[0]
            columns_changed = True
            new_columns = transformed_data.columns.values
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
            training_data = training_data.drop(columns=target_column)
            training_data_pre_feat_selection = training_data_pre_feat_selection.drop(columns=[target_column])

            # transformed_data = transformed_data[training_data_pre_feat_selection.columns]

            pca = transform_obj
            transformed_data = pca.transform(transformed_data)


        print(f"Transformation: {transform_name}")
        print(f"Type of transformed_data: {type(transformed_data)}")
        if isinstance(transformed_data, (pd.DataFrame, np.ndarray)):
            print(f"Shape of transformed_data: {transformed_data.shape}")

    data = convert_to_dataframe(transformed_data, columns=training_data_pre_feat_selection.columns, index=predict_data.index)

    if data is None:
        data = pd.DataFrame(transformed_data, columns=training_data_pre_feat_selection.columns, index=predict_data.index)
        #raise ValueError('Transformation application failed')

    final_predict_data = data.astype(float)
    print("Predict_data_post_transformations:")
    print(final_predict_data)
    return final_predict_data

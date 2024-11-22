# -*- coding: utf-8 -*-
"""
Created on: 10/3/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
import joblib
import numpy as np
import pandas as pd
from caf.ml.functions.process_data_functions import (convert_to_dataframe,
                                                     process_data_numeric,
                                                     index_sorter,
                                                     function_remove_spaces,
                                                     find_numeric_target_column,
                                                     drop_rows)
from caf.ml.functions.NorCom_caf_ml.larger_cafml_model.norcom_inputs import Models, Default_regression_methods, Models_List_
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import (TimeSeriesSplit,
                                     cross_val_score,
                                     train_test_split, RandomizedSearchCV, StratifiedShuffleSplit)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import time


def norcom_run_functions(params, training, test, validation):
    start_time = time.time()
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

    dat, transformations = pre_forecast_data_analysis_modified(data=training,
                                                              regression_method=model,
                                                              target_column=params.target_column,
                                                              threshold=None,
                                                              output_folder=params.output_folder,
                                                              features_to_transform=None,
                                                              weight_column=params.weight_column,
                                                              index_columns=params.index_columns)

    # output_path = os.path.join(params.output_folder, 'final_training_data.csv')
    # if os.path.exists(output_path):
    #     print('final_training_data exists so is being read in')
    #     final_training = pd.read_csv(output_path)
    #     if all(col in final_training.columns for col in params.index_columns):
    #         final_training.set_index(params.index_columns)
    # else:
        # final_training = quick_feature_selection(data=df, target_column=params.target_column, cv=5, regression_method=model, output_folder=params.output_folder)
        # final_training = simple_feature_selection_test(data=df, target_column=params.target_column,
        #                                                cv=5, regression_method=model,
        #                                                index_columns=params.index_columns, weight_column=params.weight_column,
        #                                                output_folder=params.output_folder)

        # hs_output_path = os.path.join(params.output_folder, 'hybrid_sample.csv')
        # if os.path.exists(output_path):
        #     hybrid_sample = pd.read_csv(hs_output_path)
        #     if all(col in hybrid_sample.columns for col in params.index_columns):
        #         hybrid_sample.set_index(params.index_columns)
        # else:
        #     hybrid_sample = create_complex_hybrid_sample(df=dat,
        #                                                  output_folder=params.output_folder,
        #                                                  target_column=params.target_column)

        # final_training_sample, final_training = feat_selec_feat_importance(df=hybrid_sample,
        #                                                                    model=model,
        #                                                                    target_column=params.target_column,
        #                                                                    weight_column=params.weight_column,
        #                                                                    cv_folds=5,
        #                                                                    output_folder=params.output_folder,
        #                                                                    dat=dat)

        # final_training.to_csv(output_path, index=True)


    hyperparameters, best_model = modified_hyper_optimisation(regression_method=model,
                                                              data=dat,
                                                              target_column=params.target_column,
                                                              output_folder=params.output_folder)

    # save_model_and_parameters(regression_method=model,
    #                          hyperparameters=hyperparameters,
    #                          transformations=transformations,
    #                          output_folder=params.output_folder,
    #                          skip_data_analysis=None,
    #                          skip_hyperparameter_optimisation=None)


    test = apply_transformations_norcom(predict_data=test,
                                       transformations=transformations,
                                       target_column=params.target_column,
                                       numerical_features=params.numerical_features,
                                       categorical_features=params.categorical_features,
                                       output_folder=params.output_folder,
                                       training_data=dat,
                                       features_to_transform=None,
                                       training_data_pre_feat_selection=dat)

    final_predict_data = align_dataframes(df1=dat, df2=test,
                                          output_folder=params.output_folder)

    predictions, y_proba = predict_refined(trained_data=dat,
                                           predict_data=final_predict_data,
                                           trained_model=model,
                                           target_column=params.target_column,
                                           output_folder=params.output_folder,
                                           FinalModelParameters=hyperparameters,
                                           index_col=params.index_columns)


    simple_eval_model(validation_df=validation,
                      y_pred=predictions,
                      target_column=params.target_column,
                      output_folder=params.output_folder)

    end_time = time.time()
    print(f"Total run time: {end_time - start_time:.2f} seconds")

    return


"""def norcom_feature_selection(data, model, cv_method, splits, repeats, target_column,  multiple_year_prediction, categorical_data):
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
"""


# def quick_feature_selection(data, target_column, cv, regression_method, output_folder):
#     print(regression_method)
#
#     if isinstance(regression_method, LogisticRegression):
#         regression_method.set_params(max_iter=1000)
#
#     print('Feature selection beginning')
#     X = data.drop(columns=[target_column])
#     y = data[target_column]
#
#     rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#
#     # tqdm progress bar
#     with tqdm(total=1, desc="Fitting RandomForest") as pbar:
#         rf.fit(X, y)
#         pbar.update(1)
#
#     # SelectFromModel to select features
#     selector = SelectFromModel(rf, prefit=True)
#     selected_features = X.columns[selector.get_support()].tolist()
#
#
#     # tqdm progress bar for cv
#     with tqdm(total=cv, desc="Cross-validation") as pbar:
#         final_score = cross_val_score(regression_method, X[selected_features], y, cv=cv, scoring='accuracy',
#                                       n_jobs=-1, verbose=0)
#         pbar.update(cv)
#
#     print(f"Number of features selected: {len(selected_features)}")
#     print(f"Cross-validated ROC AUC score: {final_score.mean()}")
#
#     if final_score.mean() < 0.55:
#         print('ROC AUC score too low, omitting feature selection.')
#         return data
#
#     dataframe_final = pd.concat([X[selected_features], y], axis=1)
#     print(dataframe_final)
#     print(dataframe_final.shape)
#     return dataframe_final


def simple_feature_selection_test(data,
                                  target_column,
                                  cv,
                                  regression_method,
                                  output_folder,
                                  weight_column,
                                  index_columns):
    print('Feature selection beginning')

    if isinstance(regression_method, LogisticRegression):
        regression_method.set_params(max_iter=1000)

    weight_series = data[weight_column]
    weight = weight_series.values.flatten()
    data = data.drop(columns=weight_column)

    X = data.drop(columns=[target_column])
    y = data[target_column]


    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    with tqdm(total=1, desc="Fitting RandomForest") as pbar:
        rf.fit(X, y, sample_weight=weight)
        pbar.update(1)

    selector = SelectFromModel(rf, prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()

    if hasattr(cv, 'n_splits'):
        num_folds = cv.n_splits
    else:
        num_folds = cv

    fold_scores = []
    with tqdm(total=num_folds, desc="Cross-validation") as pbar:
        for fold in range(num_folds):
            score = cross_val_score(regression_method,
                                    X[selected_features],
                                    y,
                                    cv=cv,
                                    scoring='accuracy',
                                    n_jobs=-1,
                                    verbose=0)[0]
            fold_scores.append(score)
            pbar.update(1)

    mean_score = sum(fold_scores) / num_folds
    std_score = (sum((x - mean_score) ** 2 for x in fold_scores) / num_folds) ** 0.5

    print(f"Number of features selected: {len(selected_features)}")
    print(f"Cross-validated ROC AUC score: {mean_score:.3f} (+/- {std_score:.3f})")

    dataframe_final = pd.concat([X[selected_features], y], axis=1)
    dataframe_final.set_index(data.index)
    if weight_series is not None:
        dataframe_final[weight_column] = weight_series
    print('Feature selection finished')

    return dataframe_final


def feat_selec_feat_importance(output_folder,
                               dat,
                               df,
                               model,
                               target_column,
                               weight_column,
                               cv_folds=5,
                               ):

    print('feature selection function beginning')
    if isinstance(model, LogisticRegression):
        model.set_params(max_iter=4000)
    model_file = os.path.join(output_folder, 'model_fit_on_sample_of_dataset.pkl')
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print('Loaded existing sample fitted model.')
    else:
        x = df.drop(columns=[target_column, weight_column], errors='ignore')
        y = df[target_column]
        print('fitting model on sample dataset')
        model.fit(x, y)
        joblib.dump(model, model_file)
        print('Fitted and saved model.')

    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        weight_series = df[weight_column]
        target_series = df[target_column]
        features_df = df.drop(columns=[target_column, weight_column], errors='ignore')

        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
        else:
            importance_values = np.abs(model.coef_[0])

        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        feature_importance.to_csv(os.path.join(output_folder, 'feature_importances.csv'),
                                  index=False)

        best_score = 0
        best_n = 0
        for top_n in tqdm(range(1, len(feature_importance) + 1), desc="Evaluating features with feature importance"):
            selected_features = feature_importance.head(top_n)['feature'].tolist()
            x = features_df[selected_features]
            y = df[target_column]

            # Perform cross-validation
            scores = cross_val_score(model, x, y, cv=cv_folds, n_jobs=-1)
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_n = top_n

        best_features = feature_importance.head(best_n)['feature'].tolist()
        print(best_features)
        print(best_score)
        selected_df = df[best_features + weight_series, target_series]
        selected_df = pd.concat([selected_df, target_series], axis=1)
        selected_df = pd.concat([selected_df, weight_series], axis=1)
        selected_df.set_index(df.index)

        selected_training_data = dat[best_features + [target_column, weight_column]]
        selected_training_data.set_index(dat.index, inplace=True)

        print('feature selection function ending')
        return selected_df
    else:
        print('Model does not have feature importances. Select model that does. \
               Feature selection is being skipped.')
        return df


def modified_hyper_optimisation(regression_method, data, target_column, output_folder):
    print('Hyperparameter optimisation underway')
    model_filename = os.path.join(output_folder, 'best_model.pkl')
    params_filename = os.path.join(output_folder, 'best_params.pkl')

    if os.path.exists(model_filename):
        model_fit = joblib.load(model_filename)
        best_params = joblib.load(model_filename)
        return best_params, model_fit

    x = data.drop(columns=[target_column])
    y = data[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    cv = TimeSeriesSplit(n_splits=3)

    def _grid_search(model_instance,
                     param_grid,
                     cv,
                     scoring):
        grid_search = RandomizedSearchCV(model_instance,
                                         param_grid,
                                         cv=cv,
                                         scoring=scoring,
                                         verbose=2,
                                         n_jobs=-1)
        grid_search.fit(x_train, y_train)
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
            param_grid = {"C": [1.0, 0.1, 0.01], "l1_ratio": [0.1, 0.5, 0.9]}
        elif regression_method.penalty == 'l1':
            print('L1 parameter grid selected')
            param_grid = {"C": [1.0, 0.1, 0.01]}
        elif regression_method.penalty == 'l2':
            print('L2 parameter grid selected')
            param_grid = {"C": [1.0, 0.1, 0.01]}


    best_params = _grid_search(regression_method, param_grid, cv=cv, scoring="accuracy")
    best_model = regression_method.set_params(**best_params)
    best_model.fit(x_train, y_train)
    joblib.dump(best_model, model_filename)
    joblib.dump(best_params, params_filename)

    # Evaluate on the test set
    test_score = best_model.score(x_test, y_test)
    print(f'Test score: {test_score}')

    # Extract coefficients
    feature_names = x.columns.tolist()
    coefficients = best_model.coef_[0]
    intercept = best_model.intercept_[0]
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['Intercept'] = intercept

    coef_df.to_csv(os.path.join(output_folder, 'coefficients_from_trained_model.csv'), index=True)

    return best_params, best_model


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

            data_encoded = pd.get_dummies(transformed_data, columns=transformed_data.columns, drop_first=True,
                                          dtype=float)

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


def align_dataframes(df1, df2, output_folder):
    common_columns = df1.columns.intersection(df2.columns)
    aligned_df2 = df2[common_columns]

    predict_data_path = os.path.join(output_folder, 'Final_prediction_data.csv')

    if os.path.exists(predict_data_path):
        print(
            f"The file Final_prediction_data.csv already exists in {output_folder} and is being replaced.")

    aligned_df2.to_csv(predict_data_path, index=True)
    print(f"Predict data post transformations saved to: {predict_data_path}")

    return aligned_df2


def predict_refined(trained_data, predict_data, trained_model, target_column,
                    output_folder, index_col, FinalModelParameters):
    # Handle missing columns
    common_columns = set(trained_data.columns) & set(predict_data.columns)
    common_columns = list(common_columns - {target_column})

    x_train = trained_data[common_columns]
    y_train = trained_data[target_column]
    x_predict = predict_data[common_columns]

    # Set hyperparameters
    if FinalModelParameters:
        trained_model.set_params(**FinalModelParameters)
        print("Using provided hyperparameters:")
        print(trained_model.get_params())
    else:
        print("Using default model parameters")


    trained_model.fit(x_train, y_train)
    predictions = trained_model.predict(x_predict)
    y_proba = trained_model.predict_proba(x_predict)

    # Save predictions
    prediction_df = pd.DataFrame({target_column: predictions}, index=predict_data.index)
    prediction_file_path = os.path.join(output_folder, 'predictions.csv')
    prediction_df.to_csv(prediction_file_path, index_label=index_col, index=True)
    print(f"Predictions saved to: {prediction_file_path}")

    return predictions, y_proba


def experimental_functions(data, categorical_transformations, features_to_interact, features_to_transform, output_folder):
    transformations = []
    transformations.extend(categorical_transformations)

    if isinstance(features_to_interact, pd.Index):
        features_to_interact = list(features_to_interact)

    if len(features_to_interact) > 10:
        features_to_interact = features_to_interact[:10]

    interaction_terms = {}
    for i, f1 in enumerate(features_to_interact):
        for f2 in features_to_interact[i + 1:]:
            interaction_terms[f'{f1}_{f2}_interaction'] = data[f1] * data[f2]

    interaction_df = pd.DataFrame(interaction_terms)
    final_df = pd.concat([data, interaction_df], axis=1)

    if features_to_transform is None or len(features_to_transform) == 0:
        features_to_transform = data.columns.tolist()

    poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=True)
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


def create_simple_hybrid_sample(df, output_folder):
    print('Dataframe size before reduction')
    print(df.shape)
    sample_fraction = 0.01

    # Random sampling
    sampled_df = df.sample(frac=sample_fraction, random_state=42)

    print('Dataframe size post reduction')
    print(sampled_df.shape)

    output_filename = 'hybrid_sample.csv'
    output_path = os.path.join(output_folder, output_filename)
    sampled_df.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"Reduced size data exported to: {output_path}")

    return sampled_df


def create_complex_hybrid_sample(df, target_column, output_folder):
    print('Reducing dataframe size to improve run time')
    sample_fraction = 0.02
    x = df.drop(columns=[target_column])
    y = df[target_column]

    sampled_df = None
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=sample_fraction, random_state=42)
    for train_index, sample_index in strat_split.split(x, y):
        sampled_df = df.loc[sample_index]


    output_filename = 'hybrid_sample.csv'
    output_path = os.path.join(output_folder, output_filename)
    sampled_df.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"Reduced size data exported to: {output_path}")

    return sampled_df


# Example usage:
# df = pd.read_csv('your_large_dataset.csv')
# target_column = 'target'
# output_folder = './output'
# sampled_df = create_hybrid_sample(df, target_column, output_folder, sample_fraction=0.01)


###############################################################################


def refined_data_processor_function(df,
                                    target_column,
                                    index_columns,
                                    categorical_features,
                                    numerical_features,
                                    output_folder,
                                    column_name_to_drop_rows,
                                    value_in_row,
                                    weight_column
                                    ):
    """
    :param df: Input data that is taken directly from a path provided in the
               run file. Classified build should be default (cb_tfn_v15)
    :param target_column: This is a column in the df dataframe that is being
                          predicted. The column should be categorical and
                          for NorCom is 'numcarvan'.
    :param index_columns: These are columns in the df dataframe that should be
                          indexed. They should be relevant to the data but not
                          to the modelling process. This is the segmentation of
                          the data.
    :param categorical_features: These are columns in the df dataframe that
                                 are categorical.
    :param numerical_features: These are columns in the df dataframe that
                               are continuous.
    :param output_folder: This is a path to a folder where all outputs should
                          be written.
    :param column_name_to_drop_rows: LEFT AS DEFAULT (see run rile).
    :param value_in_row: LEFT AS DEFAULT (see run rile).
    :param weight_column: This is a column in the df dataframe that acts as
                          a weight variable for missing trips.
    :return: Returns tidied data ready for the next phase of data processing.
             This preparation ensures the classified build is ready to be
             modelled.


    The function begins by identifying which columns in the classified build to
    keep. The following array of functions are taken directly from the
    caf.ml repository. process_data_numeric ensures all data is numeric in the
    processed dataframe. index_sorter sets the index columns as an index or
    multiindex. function_remove_spaces removed any whitespaces.
    find_numeric_target_column checks that the specified target column is
    numeric and therefore can be modelled. drop_rows removed any specified
    rows (see run_file for default). convert_to_dataframe converts the
    data to a dataframe for the next steps of modelling. The dataframe at this
    point is then written to the output folder provided as a csv file.

    """
    df = pd.read_csv(df, low_memory=False)
    new_df = df[index_columns]
    new_df.to_csv(os.path.join(output_folder, 'index_columns_csv.csv'), index=False)
    target_column_ = None
    weight_column_ = None
    if isinstance(target_column, str):
        target_column_ = [target_column]
    if isinstance(weight_column, str):
        weight_column_ = [weight_column]

    if numerical_features is None:
        columns_to_keep = index_columns + categorical_features + target_column_ + weight_column_
    elif categorical_features is None:
        columns_to_keep = index_columns + numerical_features + target_column_ + weight_column_
        print(columns_to_keep)
    else:
        columns_to_keep = index_columns + categorical_features + numerical_features + target_column_ + weight_column_
    df1 = df[columns_to_keep]
    x_ = process_data_numeric(df1, keep_columns=None)
    data = index_sorter(x_, index_columns=index_columns, drop_columns=None)
    final_data = function_remove_spaces(data)
    final_data = convert_to_dataframe(final_data)

    final_data = find_numeric_target_column(final_data, target_column=target_column, categorical_target=None)
    final_data = drop_rows(final_data,
                           column_name_to_drop_rows=column_name_to_drop_rows,
                           value_in_row=value_in_row)

    final_data = convert_to_dataframe(final_data)
    output_filename = 'initial_processed_data.csv'
    output_path = os.path.join(output_folder, output_filename)
    final_data.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"initial_processed_data exported to: {output_path}")
    return final_data, new_df


def encode_and_sort(df,
                    target_column,
                    output_folder,
                    categorical_feat,
                    training_year,
                    weight_column,
                    binary_prediction):
    """
    :param df: This is the input data and is set to the output from the
               refined_data_processor_function.
    :param target_column: see refined_data_processor_function documentation.
    :param output_folder: see refined_data_processor_function documentation.
    :param categorical_feat: see refined_data_processor_function documentation.
    :param training_year: This is set outside the model as an integer
                          and is used as a splitting point in the time series
                          data. Data before this point is
    :param weight_column: see refined_data_processor_function documentation.
    :param binary_prediction:
    :return: returns the final data split into three separate dataframes that
             represent training, test and validation. Training is what the
             model is trained on. Test is left as unseen and the trained
             model predicts on this data. Validation is truth data and used
             to evaluate the predictions.
    """
    df = df.apply(pd.to_numeric, errors='coerce')

    if target_column in df.columns:
        x = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        x = df
        y = None

    data_encoded = pd.get_dummies(x, columns=categorical_feat, drop_first=True, dtype=float)

    if y is not None:
        data_encoded[target_column] = y

    df.loc[:, target_column] = df[target_column].astype(int)

    training_df = data_encoded.loc[df.index.get_level_values('surveyyear') <= int(training_year)]
    test_df = data_encoded.loc[df.index.get_level_values('surveyyear') > int(training_year)]
    test_df = test_df.drop(columns=weight_column)

    if target_column in test_df.columns:
        validation_df = test_df[[target_column]]
    else:
        raise ValueError('Check test dataframe for target column')

    if binary_prediction == '0vs1':
        print('0 vs 1 model selected')
        for df in [training_df, validation_df]:
            df = df[df[target_column].isin([0, 1])]
            df.loc[:, target_column] = df[target_column].astype(int)

    if binary_prediction == '1vs2':
        print('1 vs 2 model selected')
        for df in [training_df, validation_df]:
            df = df[df[target_column].isin([1, 2])]
            df.loc[:, target_column] = df[target_column].astype(int)

    if binary_prediction is None:
        for df in [training_df, validation_df]:
            df[target_column] = df[target_column].apply(lambda x: x if x in [0, 1] else 2)
            df.loc[:, target_column] = df[target_column].astype(int)

    training_df.to_csv(os.path.join(output_folder, 'training_data.csv'))
    test_df.to_csv(os.path.join(output_folder, 'test_data.csv'))
    validation_df.to_csv(os.path.join(output_folder, 'validation_data.csv'), index=True)

    return training_df, test_df, validation_df


def pre_forecast_data_analysis_modified(data,
                                        regression_method,
                                        target_column,
                                        threshold,
                                        output_folder,
                                        features_to_transform,
                                        weight_column,
                                        index_columns):
    dataframe_final = None
    transformations = []
    da_output_path = os.path.join(output_folder, 'df_post_data_analysis.csv')
    transformations_file_path = output_folder / 'transformations.pkl'
    if os.path.exists(da_output_path):
        print('df_post_data_analysis exists so is being read in')
        dataframe_final = pd.read_csv(da_output_path)
        if all(col in dataframe_final.columns for col in index_columns):
            dataframe_final.set_index(index_columns)
        if weight_column not in dataframe_final.columns:
            weight_series = data[weight_column]
            dataframe_final[weight_column] = weight_series
        transformations = joblib.load(transformations_file_path)
        return dataframe_final, transformations
    else:

        print(regression_method)
        alpha = 0.05
        weight_series = None
        if weight_column in data.columns:
            weight_series = data[weight_column]
            data = data.drop(columns=weight_column)
        target_series = data[target_column]

        x_ = data.drop(columns=[target_column])
        y = data[target_column]
        y = pd.to_numeric(y, errors='coerce')
        x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.2, random_state=42)

        any_issue_present = False

        saved_model_path = os.path.join(output_folder, 'model_for_analysis.pkl')
        if os.path.exists(saved_model_path):
            model_fit = joblib.load(saved_model_path)
        else:
            model_fit = regression_method.fit(x_train, y_train)
            joblib.dump(model_fit, saved_model_path)

        y_pred = model_fit.predict(x_test)
        y_pred = pd.to_numeric(y_pred, errors='coerce')
        residuals = y_test - y_pred

        # Linearity
        print('Checking linearity')
        for col in x_test.columns:
            correlation = np.corrcoef(x_test[col], residuals)[0, 1]
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
        X_with_const = add_constant(x_test)
        bp_test_statistic, bp_test_p_value, _, _ = het_breuschpagan(residuals, X_with_const)
        print(f"Breusch-Pagan test p-value: {bp_test_p_value}")
        if bp_test_p_value < alpha:
            print("Warning: Breusch-Pagan test suggests heteroscedasticity.")
            any_issue_present = True

        print(any_issue_present)
        if any_issue_present:
            # transformed_data = x_.map(lambda x: np.log(x + 1))
            # transformations.append(('log', None))

            # df_final_to_model, transformations = experimental_functions(data=transformed_data,
            #                                                             categorical_transformations=transformations,
            #                                                             features_to_interact=x_.columns,
            #                                                             features_to_transform=features_to_transform,
            #                                                             output_folder=output_folder)

            # numerical_pipeline = Pipeline([
            #     ('imputer', SimpleImputer(strategy='median')),
            #     ('scaler', StandardScaler())
            # ])
            # numerical_data = numerical_pipeline.fit_transform(transformed_data)
            # transformations.append(('scaling', None))

            fa = FactorAnalysis()
            dataframe = fa.fit_transform(x_)
            transformations.append(('FA', fa))

            # pca = PCA()
            # dataframe = pca.fit_transform(numerical_data)
            # transformations.append(('PCA', pca))

            dataframe = pd.DataFrame(dataframe, columns=x_.columns, index=data.index)

            dataframe_final = pd.concat([dataframe, y], axis=1)

            if weight_series is not None:
                dataframe_final[weight_column] = weight_series

            if output_folder is not None:
                output_filename = 'df_post_data_analysis.csv'
                output_path = os.path.join(output_folder, output_filename)
                dataframe_final.to_csv(output_path, index=True)
                print(f"data_analysis_dataframe: {output_path}")

            print("data_analysis_dataframe:")
            print(dataframe_final)
            joblib.dump(transformations, transformations_file_path)

        return dataframe_final, transformations


def simple_eval_model(
                      validation_df,
                      y_pred,
                      target_column,
                      output_folder):

    y_truth = validation_df[target_column]

    precision, recall, fscore, _ = precision_recall_fscore_support(y_truth,
                                                                   y_pred,
                                                                   average='weighted')
    metrics_dict = {
        'Precision': precision,
        'Recall': recall,
        'F1-score': fscore
    }

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(os.path.join(output_folder, 'model_evaluation_metrics.csv'), index=False)

    # Feat importance
    # if hasattr(model, "feature_importances_"):
    #   feature_importance = pd.DataFrame({
    #        'feature': training_df.drop(columns=[target_column]).columns,
    #        'importance': model.feature_importances_
    #    }).sort_values('importance', ascending=False)

    #    feature_importance.to_csv(os.path.join(output_folder, 'feature_importances.csv'),
    #                              index=False)

    return

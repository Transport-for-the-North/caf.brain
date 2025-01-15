# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 10/10/2024
Original author: Adil Zaheer
"""
import gc
import os
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, make_scorer, \
    roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV,
                                     train_test_split,
                                     cross_val_score,
                                     TimeSeriesSplit,
                                     )
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from tqdm import tqdm
from caf.ml.CODE_OVERHAUL.process_data_functions import (process_data_numeric,
                                                         index_sorter,
                                                         function_remove_spaces,
                                                         convert_to_dataframe,
                                                         find_numeric_target_column,
                                                         drop_rows)

from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import RFE
from caf.ml.functions_all.NorCom_caf_ml.probability_model.inputs import ModelStorage, ParamGridStorage
from multiprocessing import cpu_count


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
    keep. The following array of functions_all are taken directly from the
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


def transform_target_column(df, target_column, binary_prediction):
    if binary_prediction == '0vs1':
        print('0 vs 1 model selected')
        df = df[df[target_column].isin([0, 1])]
    if binary_prediction == '1vs2':
        print('1 vs 2 model selected')
        # df = df[df[target_column] != 0]
        df = df[df[target_column].isin([1, 2])]
        # df[target_column] = df[target_column].apply(lambda x: 2 if x > 2 else x)
    if binary_prediction is None:
        print('0 vs 1 vs 2+ model selected')
        df[target_column] = df[target_column].apply(lambda x: x if x in [0, 1] else 2)


    df[target_column] = df[target_column].astype(int)
    unique_values = df[target_column].unique()
    print(f"Unique values in {target_column} after transformation: {unique_values}")

    return df


def encode_and_sort(df, target_column, output_folder, categorical_feat, training_year, weight_column, binary_prediction):
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
    df = transform_target_column(df=df, target_column=target_column, binary_prediction=binary_prediction)

    if target_column in df.columns:
        x = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        x = df
        y = None

    data_encoded = pd.get_dummies(x, columns=categorical_feat, drop_first=True, dtype=float)

    if y is not None:
        data_encoded[target_column] = y

    training_df = data_encoded.loc[df.index.get_level_values('surveyyear') <= int(training_year)]
    test_df = data_encoded.loc[df.index.get_level_values('surveyyear') > int(training_year)]
    test_df = test_df.drop(columns=weight_column)

    if target_column in test_df.columns:
        validation_df = test_df[[target_column]]
    else:
        raise ValueError('Check test dataframe for target column')

    training_df.to_csv(os.path.join(output_folder, 'training_data.csv'))
    test_df.to_csv(os.path.join(output_folder, 'test_data.csv'))
    validation_df.to_csv(os.path.join(output_folder, 'validation_data.csv'), index=True)

    print(training_df)
    print(test_df)
    print(validation_df)
    return training_df, test_df, validation_df


def generate_stats_model(training_df,
                         test_df,
                         validation_df,
                         target_column,
                         output_folder):
    for df in [training_df, validation_df]:
        df[target_column] = df[target_column].apply(lambda x: x if x in [0, 1] else 2)
        df[target_column] = df[target_column].astype(int)


    X_train = training_df.drop(target_column, axis=1)
    y_train = training_df[target_column]
    X_test = test_df.drop(target_column, axis=1)


    model_filename = os.path.join(output_folder, 'stats_model.pkl')
    if os.path.exists(model_filename):
        print(f"Loading existing model from: {model_filename}")
        result = joblib.load(model_filename)
    else:
        X_train_sm = sm.add_constant(X_train)
        multinomial_model = sm.MNLogit(y_train, X_train_sm)
        result = multinomial_model.fit(disp=0, max_iter=1000, method='newton')
        joblib.dump(result, model_filename)
        print(f"Model saved to: {model_filename}")

    print(result.summary())
    summary_df = pd.read_html(result.summary().tables[1].as_html(), header=0, index_col=0)[0]
    summary_df.to_csv(os.path.join(output_folder, 'model_summary.csv'))

    X_test_sm = sm.add_constant(X_test)
    predictions = result.predict(X_test_sm)

    pred_classes = np.argmax(predictions, axis=1)
    y_true = validation_df[target_column].values

    accuracy = accuracy_score(y_true, pred_classes)
    print(f'Accuracy: {accuracy}')
    accuracy_df = pd.DataFrame({'accuracy': [accuracy]})
    accuracy_df.to_csv(os.path.join(output_folder, 'accuracy.csv'))

    final_predictions = pd.DataFrame({'predicted_target_column': pred_classes}, index=X_test.index)
    final_predictions.to_csv(os.path.join(output_folder, 'final_predictions.csv'))

    return


def generate_sklearn_model(training_df,
                           test_df,
                           validation_df,
                           target_column,
                           output_folder):
    for df in [training_df, validation_df]:
        df[target_column] = df[target_column].apply(lambda x: x if x in [0, 1] else 2)
        df[target_column] = df[target_column].astype(int)


    X_train = training_df.drop(target_column, axis=1)
    y_train = training_df[target_column]
    X_test = test_df.drop(target_column, axis=1)

    model_filename = os.path.join(output_folder, 'sklearn_model.pkl')
    if os.path.exists(model_filename):
        print(f"Loading existing model from: {model_filename}")
        best_model = joblib.load(model_filename)
    else:
        model = LogisticRegression(penalty='elasticnet', solver='saga', multi_class='multinomial',
                                   l1_ratio=0.5)

        c_values = [0.01, 0.1, 1, 10, 100]
        param_grid = {'C': c_values}

        grid_search = RandomizedSearchCV(model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, model_filename)
        print(f"Model saved to: {model_filename}")

    coefficients = best_model.coef_
    print(f"Model Coefficients: {coefficients}")

    coeff_df = pd.DataFrame(coefficients, columns=X_train.columns)
    coeff_df = coeff_df.T
    coeff_df.columns = [f'Class {i}' for i in range(coeff_df.shape[1])]
    coeff_df.reset_index(inplace=True)
    coeff_df.rename(columns={'index': 'Feature'}, inplace=True)
    coeff_df.to_csv(os.path.join(output_folder, 'elastic_net_coefficients.csv'), index=False)

    pred_probs = best_model.predict_proba(X_test)
    pred_classes = np.argmax(pred_probs, axis=1)
    y_true = validation_df[target_column].values

    accuracy = accuracy_score(y_true, pred_classes)
    print(f'Accuracy: {accuracy}')
    accuracy_df = pd.DataFrame({'accuracy': [accuracy]})
    accuracy_df.to_csv(os.path.join(output_folder, 'accuracy.csv'))

    final_predictions = pd.DataFrame({'predicted_target_column': pred_classes}, index=X_test.index)
    final_predictions.to_csv(os.path.join(output_folder, 'final_predictions.csv'))

    return


def generate_svm(training_df,
                 test_df,
                 validation_df,
                 target_column,
                 output_folder):
    for df in [training_df, validation_df]:
        df[target_column] = df[target_column].apply(lambda x: x if x in [0, 1] else 2)
        df[target_column] = df[target_column].astype(int)

    X_train = training_df.drop(target_column, axis=1)
    y_train = training_df[target_column]
    X_test = test_df.drop(target_column, axis=1)

    model_filename = os.path.join(output_folder, 'linear_svm_model.pkl')
    if os.path.exists(model_filename):
        print(f"Loading existing model from: {model_filename}")
        best_model = joblib.load(model_filename)
    else:
        model = OneVsRestClassifier(LinearSVC())

        param_grid = {
            'estimator__C': [0.1, 1, 10],
            'estimator__loss': ['hinge', 'squared_hinge'],
        }

        grid_search = RandomizedSearchCV(model,
                                         param_distributions=param_grid,
                                         n_iter=50,
                                         cv=5,
                                         n_jobs=-1,
                                         verbose=1)

        for _ in tqdm(range(1), desc="Training SVM Model"):
            grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        joblib.dump(best_model, model_filename)
        print(f"Model saved to: {model_filename}")

    coefficients = [estimator.coef_.flatten() for estimator in best_model.estimators_]
    if coefficients:
        print(f"Model Coefficients: {coefficients}")
    else:
        print("No coefficients available for the estimators.")

    print(f"Model Coefficients: {coefficients}")

    coeff_df = pd.DataFrame(coefficients, columns=X_train.columns)
    coeff_df = coeff_df.T
    coeff_df.columns = [f'Class {i}' for i in range(coeff_df.shape[1])] # changed from 0 to 1
    coeff_df.reset_index(inplace=True)
    coeff_df.rename(columns={'index': 'Feature'}, inplace=True)
    coeff_df.to_csv(os.path.join(output_folder, 'linear_svm_coefficients.csv'), index=False)

    pred_classes = best_model.predict(X_test)
    y_true = validation_df[target_column].values

    accuracy = accuracy_score(y_true, pred_classes)
    print(f'Accuracy: {accuracy}')
    accuracy_df = pd.DataFrame({'accuracy': [accuracy]})
    accuracy_df.to_csv(os.path.join(output_folder, 'accuracy.csv'))

    final_predictions = pd.DataFrame({'predicted_target_column': pred_classes}, index=X_test.index)
    final_predictions.to_csv(os.path.join(output_folder, 'final_predictions.csv'))

    return


def caf_ml_logit(training_df,
                 test_df,
                 validation_df,
                 target_column,
                 output_folder,
                 weight_column,
                 improve_data,
                 model_to_use,
                 index_columns,
                 skip_feature_selection,
                 index_columns_df):
    print('Running caf.ml logit')
    model, residuals = model_prep(training_df=training_df,
                                  target_column=target_column,
                                  output_folder=output_folder,
                                  weight_column=weight_column,
                                  model_to_use=model_to_use)

    any_issues_present = refined_cafml_data_analysis(data=training_df,
                                                     target_column=target_column,
                                                     residuals=residuals,
                                                     weight_column=weight_column)


    if any_issues_present is True or improve_data is not None:
        # if improved data exists its read in the function below, transformations too
        improved_data, transformations = modifying_data(data=training_df,
                                                        features_to_interact=training_df.columns,
                                                        features_to_transform=None,
                                                        output_folder=output_folder,
                                                        weight_column=weight_column,
                                                        target_column=target_column,
                                                        applying_transformations=None)

    else:
        transformations = None
        improved_data = training_df

    test_df_transformed = apply_transformations(predict_data=test_df,
                                                transformations=transformations,
                                                training_data_pre_feat_selection=improved_data,
                                                output_folder=output_folder,
                                                weight_column=weight_column,
                                                target_column=target_column,
                                                index_columns=index_columns)

    final_test_data = apply_feat_selection(trained_data=improved_data,
                                           test_data=test_df_transformed,
                                           output_folder=output_folder,
                                           target_column=target_column,
                                           index_columns=index_columns)

    model_filename = os.path.join(output_folder, 'cafml_final_model.pkl')
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
    else:
        if not improved_data.index.names == index_columns:
            improved_data = improved_data.set_index(index_columns)
        x = improved_data.drop(columns=[target_column])
        y = improved_data[target_column]
        weight_df = training_df[weight_column]
        weight = weight_df.values.flatten()
        if weight_column in x.columns:
            x = x.drop(columns=weight_column)
        if target_column in x.columns:
            x = x.drop(columns=target_column)
        model = model.fit(x, y, sample_weight=weight)
        joblib.dump(model, model_filename)

    y_pred = final_prediction(model=model,
                              data=final_test_data,
                              target_column=target_column,
                              output_folder=output_folder,
                              validation=validation_df)

    simple_eval_model(training_df=improved_data,
                      validation_df=validation_df,
                      y_pred=y_pred,
                      model=model,
                      target_column=target_column,
                      output_folder=output_folder)


    return




def generate_cafml_model(training_df,
                         test_df,
                         validation_df,
                         index_columns_df,
                         target_column,
                         output_folder,
                         weight_column,
                         improve_data,
                         model_to_use,
                         index_columns,
                         binary_prediction,
                         skip_feature_selection):

    if model_to_use in ['logit_l1', 'logit_l2']:
        return caf_ml_logit(training_df,
                            test_df,
                            validation_df,
                            target_column,
                            output_folder,
                            weight_column,
                            improve_data,
                            model_to_use,
                            index_columns,
                            skip_feature_selection,
                            index_columns_df)

    print('Cafml modelling beginning')
    model_filename = os.path.join(output_folder, 'cafml_final_model.pkl')
    if os.path.exists(model_filename):
        print(f"Loading existing model from: {model_filename}")
        best_model = joblib.load(model_filename)

        training_df_modified = pd.read_csv(os.path.join(output_folder, 'final_training_data.csv'))
        training_df_modified.set_index(index_columns)

        improved_data = pd.read_csv(os.path.join(output_folder, 'improved_data.csv'))
        improved_data.set_index(index_columns)


        transformations_filename = os.path.join(output_folder, 'transformations.pkl')
        if os.path.exists(transformations_filename):
            transformations = joblib.load(transformations_filename)
        else:
            transformations = None


    else:
        model, residuals = model_prep(training_df=training_df,
                                      target_column=target_column,
                                      output_folder=output_folder,
                                      weight_column=weight_column,
                                      model_to_use=model_to_use)

        final_training_data_path = os.path.join(output_folder, 'final_training_data.csv')
        if os.path.exists(final_training_data_path):
            training_df_modified = pd.read_csv(final_training_data_path)
            training_df_modified.set_index(index_columns, inplace=True)

            transformations_filename = os.path.join(output_folder, 'transformations.pkl')
            if os.path.exists(transformations_filename):
                transformations = joblib.load(transformations_filename)
            else:
                transformations = None

            improved_data = pd.read_csv(os.path.join(output_folder, 'improved_data.csv'))
            improved_data.set_index(index_columns)

        else:
            any_issues_present = refined_cafml_data_analysis(data=training_df,
                                                             target_column=target_column,
                                                             residuals=residuals,
                                                             weight_column=weight_column)
            print(any_issues_present)
            if any_issues_present is True or improve_data is not None:
                improved_data, transformations = modifying_data(data=training_df,
                                                                features_to_interact=training_df.columns,
                                                                features_to_transform=None,
                                                                output_folder=output_folder,
                                                                weight_column=weight_column,
                                                                target_column=target_column,
                                                                applying_transformations=None)


                if skip_feature_selection is not None:
                    best_model = modified_hyper_optimisation(model=model,
                                                             data=improved_data,
                                                             target_column=target_column,
                                                             output_folder=output_folder,
                                                             weight_column=weight_column,
                                                             original_training_data=training_df,
                                                             index_columns=index_columns)

                    improved_data.to_csv(os.path.join(output_folder, 'final_training_data.csv'), index=index_columns)

                    test_df_transformed = apply_transformations(predict_data=test_df,
                                                                transformations=transformations,
                                                                training_data_pre_feat_selection=improved_data,
                                                                output_folder=output_folder,
                                                                weight_column=weight_column,
                                                                target_column=target_column,
                                                                index_columns=index_columns)

                    final_test_data = apply_feat_selection(trained_data=improved_data,
                                                           test_data=test_df_transformed,
                                                           output_folder=output_folder,
                                                           target_column=target_column,
                                                           index_columns=index_columns)

                    y_pred = final_prediction(model=best_model,
                                              data=final_test_data,
                                              target_column=target_column,
                                              output_folder=output_folder,
                                              validation=validation_df)

                    simple_eval_model(training_df=improved_data,
                                      validation_df=validation_df,
                                      y_pred=y_pred,
                                      model=best_model,
                                      target_column=target_column,
                                      output_folder=output_folder)

                    return

                training_df_modified = simple_feature_selection_test(data=improved_data,
                                                                     target_column=target_column,
                                                                     cv=TimeSeriesSplit(n_splits=5),
                                                                     regression_method=model,
                                                                     output_folder=output_folder,
                                                                     weight_column=weight_column,
                                                                     index_columns=index_columns)

            else:

                if skip_feature_selection is not None:
                    transformations = None
                    improved_data = training_df
                    best_model = modified_hyper_optimisation(model=model,
                                                             data=improved_data,
                                                             target_column=target_column,
                                                             output_folder=output_folder,
                                                             weight_column=weight_column,
                                                             original_training_data=training_df,
                                                             index_columns=index_columns)

                    improved_data.to_csv(os.path.join(output_folder, 'final_training_data.csv'),
                                         index=index_columns)

                    test_df_transformed = apply_transformations(predict_data=test_df,
                                                                transformations=transformations,
                                                                training_data_pre_feat_selection=improved_data,
                                                                output_folder=output_folder,
                                                                weight_column=weight_column,
                                                                target_column=target_column,
                                                                index_columns=index_columns)

                    final_test_data = apply_feat_selection(trained_data=improved_data,
                                                           test_data=test_df_transformed,
                                                           output_folder=output_folder,
                                                           target_column=target_column,
                                                           index_columns=index_columns)

                    y_pred = final_prediction(model=best_model,
                                              data=final_test_data,
                                              target_column=target_column,
                                              output_folder=output_folder,
                                              validation=validation_df)

                    simple_eval_model(training_df=improved_data,
                                      validation_df=validation_df,
                                      y_pred=y_pred,
                                      model=best_model,
                                      target_column=target_column,
                                      output_folder=output_folder)

                    return

                improved_data = training_df
                improved_data.to_csv(os.path.join(output_folder, 'improved_data.csv'))
                transformations = None
                training_df_modified = feat_selection_modified(data=improved_data,
                                                               target_column=target_column,
                                                               cv=TimeSeriesSplit(n_splits=5),
                                                               regression_method=model,
                                                               output_folder=output_folder,
                                                               weight_column=weight_column,
                                                               index_columns=index_columns)

        best_model = modified_hyper_optimisation(model=model,
                                                 data=training_df_modified,
                                                 target_column=target_column,
                                                 output_folder=output_folder,
                                                 weight_column=weight_column,
                                                 original_training_data=training_df,
                                                 index_columns=index_columns)

    test_df_transformed = apply_transformations(predict_data=test_df,
                                                transformations=transformations,
                                                training_data_pre_feat_selection=improved_data,
                                                output_folder=output_folder,
                                                weight_column=weight_column,
                                                target_column=target_column,
                                                index_columns=index_columns)


    final_test_data = apply_feat_selection(trained_data=training_df_modified,
                                           test_data=test_df_transformed,
                                           output_folder=output_folder,
                                           target_column=target_column,
                                           index_columns=index_columns)

    y_pred = final_prediction(model=best_model,
                              data=final_test_data,
                              target_column=target_column,
                              output_folder=output_folder,
                              validation=validation_df)

    simple_eval_model(training_df=training_df_modified,
                      validation_df=validation_df,
                      y_pred=y_pred,
                      model=best_model,
                      target_column=target_column,
                      output_folder=output_folder)

    return


def model_prep(training_df, target_column, output_folder, weight_column, model_to_use):
    print('Model prep beginning')

    x = training_df.drop(columns=[target_column])
    y = training_df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    weight = None
    if weight_column in training_df.columns:
        weight_df = x_train[weight_column]
        weight = weight_df.values.flatten()
        weight_df.to_csv(os.path.join(output_folder, 'weight.csv'))
        x_train = x_train.drop(columns=weight_column)
        x_test = x_test.drop(columns=weight_column)

    print(x_test)
    print(x_test.columns)
    model_storage = ModelStorage()
    param_grid_storage = ParamGridStorage()

    models = {
        'gb': {'model': model_storage.gb, 'filename': 'gb_basic_cafml_modelfit.pkl',
               'params': param_grid_storage.gb_params},
        'rf': {'model': model_storage.rf, 'filename': 'rf_basic_cafml_modelfit.pkl',
               'params': param_grid_storage.rf_params},
        'dt': {'model': model_storage.dt, 'filename': 'dt_basic_cafml_modelfit.pkl',
               'params': param_grid_storage.dt_params},
        'svm': {'model': model_storage.svm, 'filename': 'svm_basic_cafml_modelfit.pkl',
                'params': param_grid_storage.svm_params},
        'logistic': {'model': LogisticRegression(penalty='elasticnet', solver='saga',
                                                 multi_class='multinomial',
                                                 l1_ratio=0.5, n_jobs=-1, max_iter=1000),
                     'filename': 'logistic_basic_cafml_modelfit.pkl',
                     'params': {'C': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}},
        'svm_binary': {'model': model_storage.svm_binary, 'filename': 'svm_binary_basic_cafml_modelfit.pkl',
                       'params': param_grid_storage.svm_binary_params},
        'logit_l1': {'model': model_storage.logit_l1, 'filename': 'logit_l1_basic_cafml_modelfit.pkl',
                     'params': param_grid_storage.logit_l1_params},
        'logit_l2': {'model': model_storage.logit_l2,
                     'filename': 'logit_l2_basic_cafml_modelfit.pkl',
                     'params': param_grid_storage.logit_l2_params},
        'logit_elastic_net': {'model': model_storage.logit_elastic_net,
                              'filename': 'logit_elastic_net_basic_cafml_modelfit.pkl',
                              'params': param_grid_storage.logit_elastic_net_params},
        'logit_multinomial': {'model': model_storage.logit_multinomial,
                              'filename': 'logit_multinomial_basic_cafml_modelfit.pkl',
                              'params': param_grid_storage.logit_multinomial_params}
    }

    if model_to_use not in models or model_to_use is None:
        print(f"Invalid model specified: {model_to_use}. Defaulting to logistic regression.")
        model_to_use = 'logistic'

    model_filename = os.path.join(output_folder, models[model_to_use]['filename'])
    if os.path.exists(model_filename):
        print(f"Loading existing {model_to_use} model")
        model = models[model_to_use]['model']
        model_fit = joblib.load(model_filename)
    else:
        model = models[model_to_use]['model']
        model_fit = model.fit(x_train, y_train, sample_weight=weight)
        joblib.dump(model_fit, model_filename)

    y_pred = model_fit.predict(x_test)
    residuals = y_test - y_pred

    if model_to_use in ['svm', 'logistic', 'logit_l1', 'logit_l2',
                        'logit_elastic_net', 'logit_multinomial']:
        if hasattr(model_fit, 'coef_'):
            coefficients = model_fit.coef_
            print(f"Model Coefficients shape: {coefficients.shape}")

            coeff_df = pd.DataFrame(coefficients, columns=x_train.columns)
            coeff_df = coeff_df.T
            coeff_df.columns = [f'Class {i}' for i in range(coeff_df.shape[1])]
            coeff_df.reset_index(inplace=True)
            coeff_df.rename(columns={'index': 'Feature'}, inplace=True)
            coeff_df.to_csv(os.path.join(output_folder, 'initial_model_coefficients.csv'),
                            index=False)


    print('Model prep ending')
    return model, residuals


def refined_cafml_data_analysis(data,
                                target_column,
                                residuals,
                                weight_column,
                                threshold=5
                                ):
    print('Data analysis process beginning')
    alpha = 0.05
    x = data.drop(columns=target_column)
    x = x.drop(columns=weight_column)
    y = data[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    any_issue_present = False

    # multicolinearity
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
    X_with_const = sm.add_constant(x_test)
    bp_test_statistic, bp_test_p_value, _, _ = het_breuschpagan(residuals, X_with_const)
    print(f"Breusch-Pagan test p-value: {bp_test_p_value}")
    if bp_test_p_value < alpha:
        print("Warning: Breusch-Pagan test suggests heteroscedasticity.")
        any_issue_present = True

    print(f'Any issue present: {any_issue_present}')
    print('Data analysis process finished')
    return any_issue_present


def refined_feature_selection(data,
                              target_column,
                              cv,
                              regression_method,
                              output_folder,
                              weight_column,
                              index_columns,
                              binary_prediction):
    print('Feature selection beginning')
    final_train_filename = os.path.join(output_folder, 'final_training_data.csv')
    if os.path.exists(final_train_filename):
        dataframe_final = pd.read_csv(final_train_filename)
        dataframe_final.set_index(index_columns)
        return dataframe_final

    if isinstance(regression_method, LogisticRegression):
        regression_method.set_params(max_iter=1000)
    X = data.drop(columns=[target_column, weight_column])
    y = data[target_column]

    weight_df = data[weight_column]
    weight = weight_df.values.flatten()

    # scoring_metric = 'roc_auc_ovr' if len(np.unique(y)) > 2 else 'roc_auc'
    n_classes = len(np.unique(y))
    if n_classes > 2:
        # scoring_metric = 'roc_auc_ovr'
        def scoring_func(y_true, y_pred):
            if y_pred.ndim > 1 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            return roc_auc_score(y_true, y_pred)
    else:
        # scoring_metric = 'roc_auc'
        def scoring_func(y_true, y_pred):
            return roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')

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

    custom_scorer = make_scorer(scoring_func, needs_proba=True)

    print(f"Performing {num_folds}-fold cross validation...")
    with tqdm(total=num_folds, desc="Cross-validation") as pbar:
        fold_scores = cross_val_score(regression_method,
                                      X[selected_features],
                                      y,
                                      cv=cv,
                                      scoring=custom_scorer,
                                      n_jobs=-1,
                                      verbose=0)
        for _ in range(num_folds):
            pbar.update(1)

    print(f"Scores for each fold: {fold_scores}")
    mean_score = fold_scores.mean()
    std_score = fold_scores.std()

    print(f"Number of features selected: {len(selected_features)}")
    print(f"Cross-validated ROC AUC score: {mean_score:.3f} (+/- {std_score:.3f})")

    dataframe_final = pd.concat([X[selected_features], y], axis=1)

    dataframe_final.to_csv(os.path.join(output_folder, 'final_training_data.csv'), index=True)

    print('Feature selection finished')
    return dataframe_final


def simple_feature_selection_test(data,
                                  target_column,
                                  cv,
                                  regression_method,
                                  output_folder,
                                  weight_column,
                                  index_columns):
    print('Feature selection beginning')
    final_train_filename = os.path.join(output_folder, 'final_training_data.csv')
    if os.path.exists(final_train_filename):
        dataframe_final = pd.read_csv(final_train_filename)
        if index_columns:
            dataframe_final.set_index(index_columns, inplace=True)
        return dataframe_final

    if isinstance(regression_method, LogisticRegression):
        regression_method.set_params(max_iter=1000)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if weight_column:
        weight_df = data[weight_column]
        weight = weight_df.values.flatten()
    else:
        weight = None

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

    num_classes = len(pd.unique(y))
    if num_classes > 2:
        scoring_func = make_scorer(accuracy_score)
    else:
        scoring_func = make_scorer(roc_auc_score)

    fold_scores = []
    with tqdm(total=num_folds, desc="Cross-validation") as pbar:
        for fold in range(num_folds):
            score = cross_val_score(regression_method,
                                    X[selected_features],
                                    y,
                                    cv=cv,
                                    scoring=scoring_func,
                                    n_jobs=-1,
                                    verbose=0)[0]
            fold_scores.append(score)
            pbar.update(1)

    mean_score = sum(fold_scores) / num_folds
    std_score = (sum((x - mean_score) ** 2 for x in fold_scores) / num_folds) ** 0.5

    print(f"Number of features selected: {len(selected_features)}")
    print(f"Cross-validated ROC AUC score: {mean_score:.3f} (+/- {std_score:.3f})")

    if mean_score < 0.6:
        print('Initial feature selection attempt was inaccurate, trying \
              alternative method')
        dataframe_final = feat_selection_modified(data=data,
                                                  target_column=target_column,
                                                  cv=cv,
                                                  regression_method=regression_method,
                                                  output_folder=output_folder,
                                                  weight_column=weight_column,
                                                  index_columns=index_columns)
    else:
        dataframe_final = pd.concat([X[selected_features], y], axis=1)
        dataframe_final.to_csv(final_train_filename, index=index_columns)
        print('Feature selection finished')

    return dataframe_final


def modified_hyper_optimisation(model, data, target_column, output_folder, weight_column,
                                original_training_data, index_columns):
    print('Hyperparameter optimisation beginning')
    if isinstance(model, LogisticRegression):
        model.set_params(max_iter=1000)

    if not data.index.names == index_columns:
        data = data.set_index(index_columns)

    if weight_column in data.columns:
        data = data.drop(columns=weight_column)

    x = data.drop(columns=[target_column])
    y = data[target_column]
    cv = TimeSeriesSplit(n_splits=3)

    weight_df = original_training_data[weight_column]
    weight = weight_df.values.flatten()

    n_cores = cpu_count()
    n_jobs = max(1, n_cores - 1)

    def rand_search(model_instance, param_grid, cv, scoring):
        rand_search = RandomizedSearchCV(model_instance,
                                         param_grid,
                                         cv=cv,
                                         scoring=scoring,
                                         verbose=2,
                                         n_jobs=n_jobs,
                                         n_iter=10,
                                         return_train_score=False,
                                         pre_dispatch='1*n_jobs')
        gc.collect()
        rand_search.fit(x, y, sample_weight=weight)
        best_params = rand_search.best_params_
        print('Best parameters for model are:')
        print(best_params)
        print('CV results:')
        print(rand_search.cv_results_)
        return best_params

    param_grid_storage = ParamGridStorage()
    print(model)
    if isinstance(model, GradientBoostingClassifier):
        param_grid = param_grid_storage.gb_params
    elif isinstance(model, RandomForestClassifier):
        param_grid = param_grid_storage.rf_params
    elif isinstance(model, DecisionTreeClassifier):
        param_grid = param_grid_storage.dt_params
    elif isinstance(model, OneVsRestClassifier) and isinstance(model.estimator, LinearSVC):
        param_grid = param_grid_storage.svm_params
    elif isinstance(model, LogisticRegression):
        if model.get_params()['penalty'] == 'l1' and model.get_params()['solver'] == 'liblinear':
            param_grid = param_grid_storage.logit_l1_params
        elif model.get_params()['penalty'] == 'l2':
            param_grid = param_grid_storage.logit_l2_params
        elif model.get_params()['penalty'] == 'elasticnet':
            param_grid = param_grid_storage.logit_elastic_net_params
        elif model.get_params()['multi_class'] == 'multinomial':
            param_grid = param_grid_storage.logit_multinomial_params
        else:
            param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                          "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                          "max_iter": [1000, 2000, 3000]}
    elif isinstance(model, LinearSVC):
        param_grid = param_grid_storage.svm_binary_params
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    best_params = rand_search(model, param_grid, cv=cv, scoring="accuracy")
    best_model = model.set_params(**best_params)
    best_model.fit(x, y, sample_weight=weight)

    model_filename = os.path.join(output_folder, 'cafml_final_model.pkl')
    joblib.dump(best_model, model_filename)
    print(f"Model saved to: {model_filename}")

    # coeffs
    if hasattr(best_model, 'coef_'):
        coefficients = best_model.coef_
        if coefficients.ndim == 1:
            coeff_df = pd.DataFrame({'Feature': x.columns, 'Coefficient': coefficients})
        else:
            coeff_df = pd.DataFrame(coefficients.T,
                                    columns=[f'Class_{i}' for i in range(coefficients.shape[0])])
            coeff_df['Feature'] = x.columns
        coeff_df.to_csv(os.path.join(output_folder, 'final_model_coefficients.csv'), index=False)
        print(f"Coefficients saved to: {os.path.join(output_folder, 'final_model_coefficients.csv')}")
    else:
        print("Model does not have coefficients attribute.")

    print('Hyperparameter optimisation finished')
    return best_model


def apply_feat_selection(trained_data, test_data, output_folder, target_column, index_columns):
    print('final_training_data')
    print(trained_data)

    print('test_data_pre_feat_selection')
    print(test_data)
    if target_column in trained_data.columns:
        trained_data = trained_data.drop(columns=target_column)

    selected_columns = trained_data.columns

    aligned_test_data = test_data[selected_columns.intersection(test_data.columns)]

    if target_column in aligned_test_data.columns:
        aligned_test_data = aligned_test_data.drop(columns=target_column)

    predict_data_path = os.path.join(output_folder, 'Final_prediction_data.csv')
    if os.path.exists(predict_data_path):
        print(
            f"The file Final_prediction_data.csv already exists in {output_folder} and is being replaced.")

    aligned_test_data.to_csv(predict_data_path, index=True)

    print('Final prediction data (post feature selection):')
    print(aligned_test_data)

    return aligned_test_data


def final_prediction(model, data, target_column, output_folder, validation, binary_prediction):
    print('Prediction beginning')
    print(data.columns)
    print(data)
    if target_column in data.columns:
        data = data.drop(columns=target_column)

    accuracy = None
    pred_classes = None
    if binary_prediction == '0vs1':
        if isinstance(model, LinearSVC):
            pred_classes = model.predict(data)
            y_true = validation[target_column].values
            accuracy = accuracy_score(y_true, pred_classes)

        else:
            pred_probs = model.predict_proba(data)
            pred_classes = np.argmax(pred_probs, axis=1)
            y_true = validation[target_column].values
            accuracy = accuracy_score(y_true, pred_classes)

    if binary_prediction == '1vs2':
        if isinstance(model, LinearSVC):
            pred_classes = model.predict(data)
            y_true = validation[target_column].values
            accuracy = accuracy_score(y_true, pred_classes)

        else:
            pred_probs = model.predict_proba(data)
            pred_classes = np.argmax(pred_probs, axis=1) + 1
            y_true = validation[target_column].values
            accuracy = accuracy_score(y_true, pred_classes)


    print(f'Accuracy: {accuracy}')
    accuracy_df = pd.DataFrame({'accuracy': [accuracy]})
    accuracy_df.to_csv(os.path.join(output_folder, 'accuracy.csv'))

    final_predictions = pd.DataFrame({'predicted_target_column': pred_classes}, index=data.index)
    final_predictions.to_csv(os.path.join(output_folder, 'final_predictions_probability.csv'))
    print('Prediction finished')
    return pred_classes


def feat_selection_modified(data,
                            target_column,
                            cv,
                            regression_method,
                            output_folder,
                            weight_column,
                            index_columns):
    final_train_filename = os.path.join(output_folder, 'final_training_data.csv')
    if os.path.exists(final_train_filename):
        dataframe_final = pd.read_csv(final_train_filename)
        dataframe_final.set_index(index_columns)
        return dataframe_final

    print('Feature selection beginning')
    x = data.drop(columns=[target_column, weight_column])
    y = data[target_column]
    original_index = x.index

    weight_df = data[weight_column]
    weight = weight_df.values.flatten()

    # Lasso
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(x, y, sample_weight=weight)
    lasso_selected = x.columns[abs(lasso.coef_) > 0].tolist()

    # Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x, y, sample_weight=weight)
    ridge_selected = x.columns[abs(ridge.coef_) > np.mean(abs(ridge.coef_))].tolist()

    # RFE + logit
    rfe = RFE(estimator=LogisticRegression(random_state=42, max_iter=2000), n_features_to_select=10)
    rfe.fit(x, y, sample_weight=weight)
    rfe_selected = x.columns[rfe.support_].tolist()

    all_selected_features = list(set(lasso_selected + ridge_selected + rfe_selected))

    scores = []

    with tqdm(total=cv.n_splits, desc="Cross-validation") as pbar:
        for train_index, test_index in cv.split(x):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            regression_method.fit(X_train[all_selected_features], y_train)
            score = regression_method.score(X_test[all_selected_features], y_test)
            scores.append(score)
            pbar.update(1)

    print(f"Number of features selected: {len(all_selected_features)}")
    print(f"Cross-validated score: {np.mean(scores)}")

    cv_score = np.mean(scores)
    if cv_score < 0.5:
        print('CV score is still not optimal, feature selection is being ignored')
        data.to_csv(os.path.join(output_folder, 'final_training_data.csv'), index=index_columns)
        return data

    dataframe_final = pd.concat([x[all_selected_features], y], axis=1)
    dataframe_final.index = original_index

    dataframe_final.to_csv(os.path.join(output_folder, 'final_training_data.csv'), index=index_columns)

    print('Feature selection finished')
    return dataframe_final


def simple_eval_model(training_df,
                      validation_df,
                      y_pred,
                      model,
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


def modifying_data(data,
                   features_to_interact,
                   features_to_transform,
                   output_folder,
                   weight_column,
                   target_column,
                   applying_transformations):

    if applying_transformations is not None:
        data_ = data.copy()
        if weight_column in data_.columns:
            data_ = data_.drop(columns=weight_column)
        data_ = data_.drop(columns=target_column)

        # interaction terms
        if isinstance(features_to_interact, pd.Index):
            features_to_interact = list(features_to_interact)

        features_to_interact = [f for f in features_to_interact if f in data_.columns]

        if len(features_to_interact) > 10:
            features_to_interact = features_to_interact[:10]

        interaction_terms = {}
        for i, f1 in enumerate(features_to_interact):
            for f2 in features_to_interact[i + 1:]:
                interaction_terms[f'{f1}_{f2}_interaction'] = data_[f1] * data_[f2]

        interaction_df = pd.DataFrame(interaction_terms)
        final_df = pd.concat([data_, interaction_df], axis=1)

        # poly
        if features_to_transform is None or len(features_to_transform) == 0:
            features_to_transform = data_.columns.tolist()

        poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=True)
        if len(features_to_transform) > 10:  # this is an arbitrary threshold, investigate further
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
        final_df = pd.DataFrame(final_df)

        # scale
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        scaled_df = numerical_pipeline.fit_transform(final_df)
        final_df_scale = pd.DataFrame(scaled_df, columns=final_df.columns)
        final_df = final_df_scale.set_index(final_df.index)

        # mca
        n_components = 2
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        mca_result = svd.fit_transform(final_df)
        explained_variance_ratio = svd.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_variance_ratio}")

        mca_features = pd.DataFrame(mca_result,
                                    columns=[f'MCA{i + 1}' for i in range(n_components)],
                                    index=final_df.index) #changed from final_df to data

        final_df = pd.concat([final_df, mca_features], axis=1)

        if weight_column in final_df.columns:
            final_df = final_df.drop(columns=weight_column)
        if target_column in final_df.columns:
            final_df = final_df.drop(columns=target_column)

        print('predict data post transformations, pre feature selection')
        print(final_df)

        return final_df, None


    else:
        _filename = os.path.join(output_folder, 'improved_data.csv')
        transformations_filename = os.path.join(output_folder, 'transformations.pkl')
        if os.path.exists(_filename):
            final_data = pd.read_csv(_filename)
            transformations = joblib.load(transformations_filename)
            return final_data, transformations


        transformations_ = []
        data_ = data.copy()
        weight_series = None
        if weight_column in data_.columns:
            weight_series = data_[weight_column]
            data_ = data_.drop(columns=weight_column)

        target_series = data_[target_column]
        data_ = data_.drop(columns=target_column)

        if isinstance(features_to_interact, pd.Index):
            features_to_interact = list(features_to_interact)

        features_to_interact = [f for f in features_to_interact if f in data_.columns]

        if len(features_to_interact) > 10:
            features_to_interact = features_to_interact[:10]


        interaction_terms = {}
        for i, f1 in enumerate(features_to_interact):
            for f2 in features_to_interact[i + 1:]:
                interaction_terms[f'{f1}_{f2}_interaction'] = data_[f1] * data_[f2]

        interaction_df = pd.DataFrame(interaction_terms)
        final_df = pd.concat([data_, interaction_df], axis=1)


        if features_to_transform is None or len(features_to_transform) == 0:
            features_to_transform = data_.columns.tolist()

        poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=True)
        if len(features_to_transform) > 10:  # this is an arbitrary threshold, investigate further
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

        # scale
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        scaled_df = numerical_pipeline.fit_transform(final_df)
        final_df = pd.DataFrame(scaled_df, columns=final_df.columns)
        final_df = final_df.set_index(final_df.index)

        # mca
        n_components = 2
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        mca_result = svd.fit_transform(final_df)
        explained_variance_ratio = svd.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_variance_ratio}")

        mca_features = pd.DataFrame(mca_result,
                                    columns=[f'MCA{i + 1}' for i in range(n_components)],
                                    index=final_df.index)
        final_df = pd.concat([final_df, mca_features], axis=1)

        final_df = final_df.set_index(data_.index)
        final_df = pd.concat([final_df, target_series], axis=1)

        if weight_series is not None:
            final_df[weight_column] = weight_series

        print(final_df)
        transformations_.append(('interaction_poly_mca', None))

        transformations_file_path = output_folder / 'transformations.pkl'
        joblib.dump(transformations_, transformations_file_path)
        print(f'Transformations saved to: {transformations_file_path}')

        final_df.to_csv(os.path.join(output_folder, 'improved_data.csv'), index=True)


        return final_df, transformations_


def apply_transformations(predict_data, transformations,
                          training_data_pre_feat_selection, output_folder, weight_column, target_column,
                          index_columns):

    if transformations is None:
        return predict_data

    if weight_column in training_data_pre_feat_selection.columns:
        training_data_pre_feat_selection = training_data_pre_feat_selection.drop(columns=weight_column)

    if all(col in training_data_pre_feat_selection.columns for col in index_columns):
        training_data_pre_feat_selection = training_data_pre_feat_selection.set_index(index_columns)
        training_data_pre_feat_selection = training_data_pre_feat_selection.drop(index_columns)

    transformed_data = predict_data.copy()

    for transform_name, transform_obj in transformations:
        if transform_name == 'interaction_poly_mca':
            transformed_data, _ = modifying_data(data=transformed_data,
                                                 features_to_interact=transformed_data.columns,
                                                 features_to_transform=None,
                                                 output_folder=output_folder,
                                                 weight_column=weight_column,
                                                 target_column=target_column,
                                                 applying_transformations='yes')

        print(f"Transformation: {transform_name}")
        print(f"Type of transformed_data: {type(transformed_data)}")
        if isinstance(transformed_data, (pd.DataFrame, np.ndarray)):
            print(f"Shape of transformed_data: {transformed_data.shape}")

    final_predict_data = transformed_data.astype(float)
    print("Predict_data_post_transformations:")
    print(final_predict_data)
    return final_predict_data

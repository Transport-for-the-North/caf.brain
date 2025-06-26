# -*- coding: utf-8 -*-
"""
Created on: 6/11/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List
import os
import joblib
from pathlib import Path
import logging
LOG = logging.getLogger(__name__)

def preprocess_numerical_data(df: pd.DataFrame,
                              numerical_features: list[str],
                              is_test_data: bool,
                              numerical_pipeline_train=None,
                              output_folder=None) -> pd.DataFrame:
    """
    Scales data via SciKitLearns standard scalar. Separates logic for train and
    test but ensures that the same transformations applied to train are applied
    to test if used in a full model run. If not used in a full model run,
    an output path must be provided to generate the training transformations
    which would then be applied to the test data when called again.

    Train example:
    is_test_data: False
    numerical_pipeline_train = None
    output_folder: Path

    Test example:
    is_test_data: True
    numerical_pipeline_train: set to value if scalar stored in memory. Else
                              leave as None.
    output_folder: One of scale_csv.csv or numerical_pipeline.pkl is to be used.
                   These will have been generated when using the function on
                   training data. If you do not have these then the function
                   needs to be used on training data first.

    :param df: Input data.
    :param numerical_features: List of string column names that are
                               continuous variables.
    :param is_test_data: Bool. Used to dictate if test data is used or not.
    :param numerical_pipeline_train: Always None unless full model flow is
                                     running. Only is relevant to test data.
                                     If None then one of numerical_pipeline.pkl
                                     or scale_csv.csv needs to be available to
                                     apply transformations to train.
    :param output_folder: Path to output folder.
    :return: Dataframe of only the scaled numerical data set to the input
             datas index. Also, the numerical pipeline information.
    """
    if is_test_data:
        LOG.info("Processing test data - attempting to apply training transformations")
        numerical_pipeline = None
        method_used = None

        if numerical_pipeline_train is not None:
            try:
                numerical_pipeline = numerical_pipeline_train
                method_used = "in-memory pipeline"
            except Exception as e:
                LOG.warning(f"Could not use in-memory pipeline: {str(e)}")

        if numerical_pipeline is None and output_folder is not None:
            numerical_pipeline_path_pkl = os.path.join(output_folder, 'numerical_pipeline.pkl')
            try:
                numerical_pipeline = joblib.load(numerical_pipeline_path_pkl)
                method_used = "pickled pipeline"
            except Exception as e:
                LOG.warning(f"Could not load pipeline from pickle: {str(e)}")

        if numerical_pipeline is None and output_folder is not None:
            scale_csv_path = os.path.join(output_folder, 'scale_csv.csv')
            try:
                scale_df = pd.read_csv(scale_csv_path, index_col=0)

                scaler = StandardScaler()
                scaler.mean_ = scale_df.loc['mean'].values
                scaler.scale_ = scale_df.loc['std'].values
                scaler.var_ = scale_df.loc['var'].values

                numerical_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', scaler)
                ])
                method_used = "CSV scaler values"
            except Exception as e:
                LOG.warning(f"Could not load scaler values from CSV: {str(e)}")

        if numerical_pipeline is None:
            raise ValueError(
                "Unable to process test data: No transformation parameters available. \
                Please ensure either the numerical_pipeline_train parameter is provided, \
                or the output_folder contains 'numerical_pipeline.pkl' or 'scale_csv.csv'."
            )

        LOG.info(f"Successfully applied training transformations to test data using {method_used}")

        numerical_data = numerical_pipeline.transform(df[numerical_features])
        numerical_df = pd.DataFrame(numerical_data, columns=numerical_features, index=df.index)

        return numerical_df

    else:
        LOG.info("Processing training data - fitting new transformations")
        if output_folder is None:
            raise ValueError(
                "Output folder must be provided for training data to save transformations")

        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        numerical_data = numerical_pipeline.fit_transform(df[numerical_features])
        numerical_df = pd.DataFrame(numerical_data, columns=numerical_features, index=df.index)

        numerical_pipeline_path_pkl = os.path.join(output_folder, 'numerical_pipeline.pkl')
        joblib.dump(numerical_pipeline, numerical_pipeline_path_pkl)
        LOG.info(f"Saved numerical pipeline to {numerical_pipeline_path_pkl}")

        scale_csv_path = os.path.join(output_folder, 'scale_csv.csv')
        scaler = numerical_pipeline.named_steps['scaler']
        scale_df = pd.DataFrame({
            'mean': scaler.mean_,
            'std': scaler.scale_,
            'var': scaler.var_
        }, index=numerical_features).T
        scale_df.to_csv(scale_csv_path)
        LOG.info(f"Saved scaler values to {scale_csv_path}")

        return numerical_df, numerical_pipeline


def preprocess_categorical_data(df: pd.DataFrame,
                                categorical_features: List[str],
                                sample_size_encode: bool,
                                select_encode_values: bool,
                                encode_values_to_drop: List[str]
                                ) -> pd.DataFrame:
    """
    Encodes categorical variables via a choice of methods. Standard
    encoding where the first in each category is dropped is default,
    see pandas.get_dummies documentation. Encoding via sample size or values
    set by the user is also possible.

    :param df: Input dataframe.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :param sample_size_encode: Optional bool. If true, the data will be split
                               based on sample size. Variables with the largest
                               sample size will be used as reference class.
    :param select_encode_values: Optional bool. If True, data is split based
                                 on custom values set by the user. Corresponds
                                 to encode_values_to_drop.
    :param encode_values_to_drop: If select_encode_values is True, then this
                                  must be a list of strings the length of
                                  categorical_features. Position one in the list
                                  will link to the first variable provided in
                                  categorical_features and so on.
    :return: categorical_df: Dataframe of encoded categorical data set
                             to the input datas index.
    :return: drop_vals: Dataframe of columns removed during the encoding process.
    """
    df.columns = df.columns.astype(str)

    if sample_size_encode is True:
        categorical_df, drop_vals = sample_size_encode_(df=df,
                                                       categorical_features=categorical_features)
        return categorical_df, drop_vals
    elif select_encode_values is True:
        categorical_df, drop_vals = custom_sample_encode(df=df,
                                                         categorical_features=categorical_features,
                                                         drop_values=encode_values_to_drop)
        return categorical_df, drop_vals
    else:
        categorical_df = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=float)
        categorical_df.columns = categorical_df.columns.str.replace('.0', '')

        cat_w_all_cols = pd.get_dummies(df, columns=categorical_features, dtype=float)
        cat_w_all_cols.columns = cat_w_all_cols.columns.str.replace('.0', '')

        extra_columns = cat_w_all_cols.columns.difference(categorical_df.columns)
        drop_vals = cat_w_all_cols[extra_columns]

    return categorical_df, drop_vals


def sample_size_encode_(df: pd.DataFrame,
                        categorical_features: List[str]):
    """
    Encodes variables based on sample size. The value that appears most often
    in each of the categorical variables is used as the reference and therefore
    dropped during encoding.

    :param df: Input dataframe.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :return: categorical_encoded: Dataframe of encoded categorical data.
    :return: dropped_df: Dataframe of columns removed during the encoding process.
    """
    modes = df[categorical_features].mode().iloc[0]

    modified_cat_features = []
    dropped_columns = []
    for feature in categorical_features:
        dummies = pd.get_dummies(
            df[feature],
            prefix=feature,
            prefix_sep='_',
            drop_first=False,
            dtype=float
        )

        mode_value = modes[feature]
        mode_column = f"{feature}_{mode_value}"

        if mode_column in dummies.columns:
            dropped_columns.append(mode_column)
            dummies = dummies.drop(columns=[mode_column])

        modified_cat_features.append(dummies)

    categorical_encoded = pd.concat(modified_cat_features, axis=1)

    # df_num = df.drop(columns=categorical_features)
    # df_encoded = pd.concat([df_num, categorical_encoded], axis=1)

    drop_df = pd.DataFrame({'Dropped_Columns': dropped_columns})
    new_columns = drop_df['Dropped_Columns'].tolist()
    dropped_df = pd.DataFrame(columns=new_columns)

    return categorical_encoded, dropped_df


def custom_sample_encode(df: pd.DataFrame,
                         categorical_features: List[str],
                         drop_values: List[str]):
    """
    Encodes variables based on user specified values. The values should be
    specified in the order that the variables are listed in categorical_features.
    These will be the values dropped when encoded and therefore used as
    the reference.

    :param df: Input dataframe.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :param drop_values: List of strings the length of categorical_features.
                        Position one in the list will link to the first
                        variable provided in categorical_features and so on.
    :return: categorical_encoded: Dataframe of encoded categorical data.
    :return: dropped_df: Dataframe of columns removed during the encoding process.
    """
    if len(categorical_features) != len(drop_values):
        LOG.error("The number of categorical features must match the number of drop values")
        raise ValueError("The number of categorical features must match the number of drop values")

    modified_cat_features = []
    dropped_values = dict(zip(categorical_features, drop_values))
    dropped_columns = []

    for feature in categorical_features:
        dummies = pd.get_dummies(
            df[feature],
            prefix=feature,
            prefix_sep='_',
            drop_first=False,
            dtype=float
        )

        drop_value = dropped_values[feature]
        drop_column = f"{feature}_{drop_value}"

        if drop_column in dummies.columns:
            dropped_columns.append(drop_column)
            dummies = dummies.drop(columns=[drop_column])

        modified_cat_features.append(dummies)

    categorical_encoded = pd.concat(modified_cat_features, axis=1)
    # df_num = df.drop(columns=categorical_features)
    # df_encoded = pd.concat([df_num, categorical_encoded], axis=1)

    drop_df = pd.DataFrame({'Dropped_Columns': dropped_columns})
    new_columns = drop_df['Dropped_Columns'].tolist()
    dropped_df = pd.DataFrame(columns=new_columns)

    return categorical_encoded, dropped_df


def encode_test_data(test_df: pd.DataFrame,
                     categorical_features: List[str],
                     train_encoded: pd.DataFrame,
                     target_column: str,
                     weight_column: str,
                     weight_df: pd.DataFrame) -> pd.DataFrame:
    """
    This encodes the test data to match the training data.

    :param test_df: Test data to be encoded.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :param train_encoded:
    :param target_column: sting column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param weight_df: The weight column in dataframe form to be added back
                      to the encoded test data.
    :return: Encoded test data.
    """

    if target_column in train_encoded.columns:
        train_encoded = train_encoded.drop(columns=[target_column])
    if weight_column in train_encoded.columns:
        train_encoded = train_encoded.drop(columns=[weight_column])

    if target_column in test_df.columns:
        test_df = test_df.drop(columns=[target_column])

    test_encoded = pd.get_dummies(
        test_df,
        columns=categorical_features,
        prefix_sep='_',
        dtype=float
    )

    test_encoded = test_encoded.loc[:, test_encoded.columns.isin(train_encoded.columns)]

    if weight_df is not None:
        test_encoded[weight_column] = weight_df

    return test_encoded

def process_data_pipeline(df: pd.DataFrame,
                          numerical_features: List[str],
                          categorical_features: List[str],
                          target_column: str,
                          weight_column: str,
                          sample_size_encode: bool,
                          select_encode_values: bool,
                          encode_values_to_drop: List[str],
                          train_encoded: None or pd.DataFrame,
                          test_data: bool,
                          numerical_pipeline,
                          output_folder: Path):
    """
    Pipeline to process input data via encoding and scaling transformations.

    :param df: input dataframe.
    :param numerical_features: List of string column names that are
                               continuous variables.
    :param categorical_features: List of string column names that are
                                 categorical variables.
    :param target_column: String column name of value to predict.
    :param weight_column: Optional string column value to be used as weight.
    :param sample_size_encode: Optional bool. If true, the data will be split
                               based on sample size. Variables with the largest
                               sample size will be used as reference class.
    :param select_encode_values: Optional bool. If True, data is split based
                                 on custom values set by the user. Corresponds
                                 to encode_values_to_drop.
    :param encode_values_to_drop: If select_encode_values is True, then this
                                  must be a list of strings the length of
                                  categorical_features. Position one in the list
                                  will link to the first variable provided in
                                  categorical_features and so on.
    :param train_encoded: Either None or an encoded and scaled train dataset.
                          This is to ensure that the corresponding test data
                          is encoding in the same way as the training data.
                          Columns in train and test must match for prediction.
    :param test_data: If True then the dataframe passed must be the test
                      data.
    :param numerical_pipeline: This is the stored numerical pipline used on
                               the training data. Can be left as None if a full
                               model run is not being complete. In that case,
                               output_folder must contain the csv or pkl
                               version of numerical_pipeline.
    :param output_folder: Path to output folder

    :return:
        (numerical_df, categorical_df, preprocessed_df): Scaled and encoded dataframe.
        drop_vals: Dataframe of columns removed during the encoding process.
        numerical_pipeline: stored transformation pipeline for continuous variables
    """
    preprocessed_df = None

    if target_column in df.columns:
        x = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        x = df
        y = None

    weight_df = None
    drop_vals = None

    if weight_column in df.columns:
        weight_df = df[weight_column]
        x = x.drop(columns=weight_column)

    # just numerical data
    if numerical_features is not None and categorical_features is None:
        if test_data is True:
            numerical_df = preprocess_numerical_data(df=x,
                                                     numerical_features=numerical_features,
                                                     is_test_data=True,
                                                     numerical_pipeline_train=numerical_pipeline,
                                                     output_folder=output_folder)
        else:
            numerical_df, numerical_pipeline = preprocess_numerical_data(df=x,
                                                                         numerical_features=numerical_features,
                                                                         is_test_data=False,
                                                                         numerical_pipeline_train=numerical_pipeline,
                                                                         output_folder=output_folder)

        if y is not None:
            numerical_df[target_column] = y

        if weight_df is not None:
            numerical_df[weight_column] = weight_df

        return numerical_df, drop_vals, numerical_pipeline

    # just categorical data
    if categorical_features is not None and numerical_features is None:
        if test_data is True:
            test_final = encode_test_data(test_df=x,
                                          categorical_features=categorical_features,
                                          train_encoded=train_encoded,
                                          target_column=target_column,
                                          weight_column=weight_column,
                                          weight_df=weight_df)
            return test_final, None

        categorical_df, drop_vals = preprocess_categorical_data(df=x,
                                                                categorical_features=categorical_features,
                                                                sample_size_encode=sample_size_encode,
                                                                select_encode_values=select_encode_values,
                                                                encode_values_to_drop=encode_values_to_drop)
        if y is not None:
            categorical_df[target_column] = y

        if weight_df is not None:
            categorical_df[weight_column] = weight_df

        return categorical_df, drop_vals, None

    # both categorical and numerical
    if numerical_features and categorical_features is not None:
        x_cat = x.drop(columns=numerical_features)
        if test_data is True:
            numerical_df = preprocess_numerical_data(df=x,
                                                     numerical_features=numerical_features,
                                                     is_test_data=True,
                                                     numerical_pipeline_train=numerical_pipeline,
                                                     output_folder=output_folder)

            test_final = encode_test_data(test_df=x_cat,
                                          categorical_features=categorical_features,
                                          train_encoded=train_encoded,
                                          target_column=target_column,
                                          weight_column=weight_column,
                                          weight_df=weight_df)

            preprocessed_df = pd.concat([numerical_df, test_final], axis=1)

            if weight_df is not None:
                preprocessed_df[weight_column] = weight_df

            return preprocessed_df, None, None

        else:
            numerical_df, numerical_pipeline = preprocess_numerical_data(df=x,
                                                                         numerical_features=numerical_features,
                                                                         is_test_data=False,
                                                                         numerical_pipeline_train=numerical_pipeline,
                                                                         output_folder=output_folder)

            categorical_df, drop_vals = preprocess_categorical_data(df=x_cat,
                                                                    categorical_features=categorical_features,
                                                                    sample_size_encode=sample_size_encode,
                                                                    select_encode_values=select_encode_values,
                                                                    encode_values_to_drop=encode_values_to_drop)
            preprocessed_df = pd.concat([numerical_df, categorical_df], axis=1)

    if y is not None:
        preprocessed_df[target_column] = y
        preprocessed_df[target_column] = preprocessed_df[target_column].astype(int)

    if weight_df is not None:
        preprocessed_df[weight_column] = weight_df

    return preprocessed_df, drop_vals, numerical_pipeline

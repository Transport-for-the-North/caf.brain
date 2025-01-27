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


def preprocess_numerical_data(df, numerical_features):
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    numerical_data = numerical_pipeline.fit_transform(df[numerical_features])
    numerical_df = pd.DataFrame(numerical_data, columns=numerical_features, index=df.index)
    return numerical_df


def preprocess_categorical_data(df, categorical_features):
    df.columns = df.columns.astype(str)
    categorical_df = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=float)
    categorical_df.columns = categorical_df.columns.str.replace('.0', '')
    return categorical_df


def process_data_pipeline(df, numerical_features, categorical_features, target_column):
    preprocessed_df = None

    if target_column in df.columns:
        x = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        x = df
        y = None

    # just numerical data
    if numerical_features is not None and categorical_features is None:
        numerical_df = preprocess_numerical_data(x, numerical_features)
        if y is not None:
            numerical_df[target_column] = y

        return numerical_df

    # just categorical data
    if categorical_features is not None and numerical_features is None:
        categorical_df = preprocess_categorical_data(x, categorical_features)
        if y is not None:
            categorical_df[target_column] = y

        return categorical_df

    # both categorical and numerical
    if numerical_features and categorical_features is not None:
        numerical_df = preprocess_numerical_data(x, numerical_features)
        x = x.drop(columns=numerical_features)
        categorical_df = preprocess_categorical_data(x, categorical_features)
        preprocessed_df = pd.concat([numerical_df, categorical_df], axis=1)

    if y is not None:
        preprocessed_df[target_column] = y
        preprocessed_df[target_column] = preprocessed_df[target_column].astype(int)

    return preprocessed_df

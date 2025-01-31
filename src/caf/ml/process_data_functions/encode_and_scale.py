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


def preprocess_categorical_data(df,
                                categorical_features,
                                sample_size_encode,
                                select_encode_values,
                                encode_values_to_drop):
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
        drop_vals = None

    return categorical_df, drop_vals


def sample_size_encode_(df, categorical_features):
    modes = df[categorical_features].mode().iloc[0]

    modified_cat_features = []

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
            dummies = dummies.drop(columns=[mode_column])

        modified_cat_features.append(dummies)

    categorical_encoded = pd.concat(modified_cat_features, axis=1)

    df_num = df.drop(columns=categorical_features)
    df_encoded = pd.concat([df_num, categorical_encoded], axis=1)

    return df_encoded, modes


def custom_sample_encode(df, categorical_features, drop_values):
    if len(categorical_features) != len(drop_values):
        raise ValueError("The number of categorical features must match the number of drop values")

    modified_cat_features = []
    dropped_values = dict(zip(categorical_features, drop_values))

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
            dummies = dummies.drop(columns=[drop_column])

        modified_cat_features.append(dummies)

    categorical_encoded = pd.concat(modified_cat_features, axis=1)
    df_num = df.drop(columns=categorical_features)
    df_encoded = pd.concat([df_num, categorical_encoded], axis=1)

    return df_encoded, dropped_values


def encode_test_data(test_df,
                     categorical_features,
                     train_encoded,
                     target_column,
                     weight_column,
                     weight_df):

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

def process_data_pipeline(df,
                          numerical_features,
                          categorical_features,
                          target_column,
                          weight_column,
                          sample_size_encode,
                          select_encode_values,
                          encode_values_to_drop,
                          train_encoded,
                          test_data):
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
        numerical_df = preprocess_numerical_data(x, numerical_features)
        if y is not None:
            numerical_df[target_column] = y

        if weight_df is not None:
            numerical_df[weight_column] = weight_df

        return numerical_df, drop_vals

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

        return categorical_df, drop_vals

    # both categorical and numerical
    if numerical_features and categorical_features is not None:
        numerical_df = preprocess_numerical_data(x, numerical_features)
        x = x.drop(columns=numerical_features)

        if test_data is True:
            test_final = encode_test_data(test_df=x,
                                          categorical_features=categorical_features,
                                          train_encoded=train_encoded,
                                          target_column=target_column,
                                          weight_column=weight_column,
                                          weight_df=weight_df)

            preprocessed_df = pd.concat([numerical_df, test_final], axis=1)

            if y is not None:
                preprocessed_df[target_column] = y
                preprocessed_df[target_column] = preprocessed_df[target_column].astype(int)

            if weight_df is not None:
                preprocessed_df[weight_column] = weight_df

            return test_final, None

        categorical_df, drop_vals = preprocess_categorical_data(df=x,
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

    return preprocessed_df, drop_vals

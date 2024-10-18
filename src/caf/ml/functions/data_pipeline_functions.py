# -*- coding: utf-8 -*-
"""
Created on: 6/11/2024
Original author: Adil Zaheer
"""
import os

# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    #df = df.astype(int)

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    categorical_data = categorical_pipeline.fit_transform(df[categorical_features])
    categorical_df = pd.DataFrame(categorical_data.toarray(), columns=categorical_pipeline.named_steps['onehot'].get_feature_names_out(categorical_features),
                                  index=df.index)

    categorical_df.columns = categorical_df.columns.str.replace('.0', '')

    return categorical_df


def process_data_pipeline(df, numerical_features, categorical_features, target_column, output_folder):

    def save_df(df_to_save, path):
        # is the index is meaningful
        if isinstance(df_to_save.index, pd.RangeIndex):
            df_to_save.to_csv(path, index=False)
        else:
            df_to_save.to_csv(path, index=True)


    transformations_ = []
    numerical_df = None
    categorical_df = None
    preprocessed_df = None


    output_filename = 'Encoded_and_scaled_data.csv'
    output_path = os.path.join(output_folder, output_filename)


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

        transformations_.append(('Scaling and encoding', None))

        if not os.path.exists(output_path):
            save_df(numerical_df, output_path)
            print('-------------------------------------------------------------')
            print(f"Encoded and scaled exported to: {output_path}")

        print("Encoded_and_scaled_data:")
        print(numerical_df.shape)
        print(numerical_df)

        return numerical_df, transformations_

    # just categorical data
    if categorical_features is not None and numerical_features is None:
        categorical_df = preprocess_categorical_data(x, categorical_features)
        if y is not None:
            categorical_df[target_column] = y

        transformations_.append(('Scaling and encoding', None))

        if not os.path.exists(output_path):
            save_df(categorical_df, output_path)
            print('-------------------------------------------------------------')
            print(f"Encoded and scaled exported to: {output_path}")


        print("Encoded_and_scaled_data:")
        print(categorical_df.shape)
        print(categorical_df)

        return categorical_df, transformations_

    # both categorical and numerical
    if numerical_features and categorical_features is not None:
        numerical_df = preprocess_numerical_data(x, numerical_features)
        categorical_df = preprocess_categorical_data(x, categorical_features)
        preprocessed_df = pd.concat([numerical_df, categorical_df], axis=1)

    if y is not None:
        preprocessed_df[target_column] = y

    if not os.path.exists(output_path):
        save_df(preprocessed_df, output_path)
        print('-------------------------------------------------------------')
        print(f"Encoded and scaled exported to: {output_path}")

    transformations_.append(('Scaling and encoding', None))

    print("Encoded_and_scaled_data:")
    print(preprocessed_df.shape)
    print(preprocessed_df)


    return preprocessed_df, transformations_

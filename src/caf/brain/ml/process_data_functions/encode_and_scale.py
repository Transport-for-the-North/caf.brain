# Built-Ins
import logging
import os
from pathlib import Path
from typing import List

# Third Party
import joblib
import pandas as pd

# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)


def preprocess_numerical_data(
    df: pd.DataFrame,
    numerical_features: list[str],
    is_test_data: bool,
    numerical_pipeline_train=None,
    output_folder=None,
) -> pd.DataFrame:
    """
    Preprocess and scale numerical data using scikit-learn's StandardScaler.

    Applies transformations for both training and test data, ensuring that test data
    uses the same transformations as training data. Saves and loads transformation
    parameters as needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    numerical_features : list of str
        List of column names for continuous variables.
    is_test_data : bool
        Whether the data is test data.
    numerical_pipeline_train : sklearn.Pipeline or None, optional
        Pre-fitted pipeline from training data, used for test data.
    output_folder : str or pathlib.Path or None, optional
        Path to output folder for saving/loading transformation parameters.

    Returns
    -------
    numerical_df : pandas.DataFrame
        Scaled numerical data.
    numerical_pipeline : sklearn.Pipeline, optional
        The fitted pipeline (only returned for training data).
    """
    if is_test_data:
        LOG.info("Processing test data - attempting to apply training transformations")
        numerical_pipeline = None
        method_used = None

        if numerical_pipeline_train is not None:
            numerical_pipeline = numerical_pipeline_train
            method_used = "in-memory pipeline"

        if numerical_pipeline is None and output_folder is not None:
            numerical_pipeline_path_pkl = os.path.join(output_folder, "numerical_pipeline.pkl")
            numerical_pipeline = joblib.load(numerical_pipeline_path_pkl)
            method_used = "pickled pipeline"

        if numerical_pipeline is None and output_folder is not None:
            scale_csv_path = os.path.join(output_folder, "scale_csv.csv")
            scale_df = pd.read_csv(scale_csv_path, index_col=0)

            scaler = StandardScaler()
            scaler.mean_ = scale_df.loc["mean"].values
            scaler.scale_ = scale_df.loc["std"].values
            scaler.var_ = scale_df.loc["var"].values

            numerical_pipeline = Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]
            )
            method_used = "CSV scaler values"

        if numerical_pipeline is None:
            raise ValueError(
                "Unable to process test data: No transformation parameters available. \
                Please ensure either the numerical_pipeline_train parameter is provided, \
                or the output_folder contains 'numerical_pipeline.pkl' or 'scale_csv.csv'."
            )

        LOG.info(
            "Successfully applied training transformations to test data using %s", method_used
        )

        numerical_data = numerical_pipeline.transform(df[numerical_features])
        numerical_df = pd.DataFrame(numerical_data, columns=numerical_features, index=df.index)

        return numerical_df

    else:
        LOG.info("Processing training data - fitting new transformations")
        if output_folder is None:
            raise ValueError(
                "Output folder must be provided for training data to save transformations"
            )

        numerical_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        numerical_data = numerical_pipeline.fit_transform(df[numerical_features])
        numerical_df = pd.DataFrame(numerical_data, columns=numerical_features, index=df.index)

        numerical_pipeline_path_pkl = os.path.join(output_folder, "numerical_pipeline.pkl")
        joblib.dump(numerical_pipeline, numerical_pipeline_path_pkl)
        LOG.info("Saved numerical pipeline to %s", numerical_pipeline_path_pkl)

        scale_csv_path = os.path.join(output_folder, "scale_csv.csv")
        scaler = numerical_pipeline.named_steps["scaler"]
        scale_df = pd.DataFrame(
            {"mean": scaler.mean_, "std": scaler.scale_, "var": scaler.var_},
            index=numerical_features,
        ).T
        scale_df.to_csv(scale_csv_path)
        LOG.info("Saved scaler values to %s", scale_csv_path)

        return numerical_df, numerical_pipeline


def preprocess_categorical_data(
    df: pd.DataFrame,
    categorical_features: List[str],
    sample_size_encode: bool,
    select_encode_values: bool,
    encode_values_to_drop: List[str],
) -> pd.DataFrame:
    """
    Encode categorical variables using various strategies.

    Supports standard encoding (drop first), encoding by sample size, or custom
    value encoding.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    categorical_features : list of str
        List of column names for categorical variables.
    sample_size_encode : bool
        If True, use sample size encoding.
    select_encode_values : bool
        If True, use custom value encoding.
    encode_values_to_drop : list of str
        Values to drop for custom encoding.

    Returns
    -------
    categorical_df : pandas.DataFrame
        Encoded categorical data.
    drop_vals : pandas.DataFrame
        Columns removed during the encoding process.
    """
    df.columns = df.columns.astype(str)

    if sample_size_encode is True:
        categorical_df, drop_vals = sample_size_encode_(
            df=df, categorical_features=categorical_features
        )
        return categorical_df, drop_vals
    elif select_encode_values is True:
        categorical_df, drop_vals = custom_sample_encode(
            df=df, categorical_features=categorical_features, drop_values=encode_values_to_drop
        )
        return categorical_df, drop_vals
    else:
        categorical_df = pd.get_dummies(
            df, columns=categorical_features, drop_first=True, dtype=float
        )
        categorical_df.columns = categorical_df.columns.str.replace(".0", "")

        cat_w_all_cols = pd.get_dummies(df, columns=categorical_features, dtype=float)
        cat_w_all_cols.columns = cat_w_all_cols.columns.str.replace(".0", "")

        extra_columns = cat_w_all_cols.columns.difference(categorical_df.columns)
        drop_vals = cat_w_all_cols[extra_columns]

    return categorical_df, drop_vals


def sample_size_encode_(df: pd.DataFrame, categorical_features: List[str]):
    """
    Encode categorical variables by dropping the most frequent value in each.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    categorical_features : list of str
        List of column names for categorical variables.

    Returns
    -------
    categorical_encoded : pandas.DataFrame
        Encoded categorical data.
    dropped_df : pandas.DataFrame
        Columns removed during the encoding process.
    """
    modes = df[categorical_features].mode().iloc[0]

    modified_cat_features = []
    dropped_columns = []
    for feature in categorical_features:
        dummies = pd.get_dummies(
            df[feature], prefix=feature, prefix_sep="_", drop_first=False, dtype=float
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

    drop_df = pd.DataFrame({"Dropped_Columns": dropped_columns})
    new_columns = drop_df["Dropped_Columns"].tolist()
    dropped_df = pd.DataFrame(columns=new_columns)

    return categorical_encoded, dropped_df


def custom_sample_encode(
    df: pd.DataFrame, categorical_features: List[str], drop_values: List[str]
):
    """
    Encode categorical variables by dropping user-specified values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    categorical_features : list of str
        List of column names for categorical variables.
    drop_values : list of str
        Values to drop for each categorical variable.

    Returns
    -------
    categorical_encoded : pandas.DataFrame
        Encoded categorical data.
    dropped_df : pandas.DataFrame
        Columns removed during the encoding process.
    """
    if len(categorical_features) != len(drop_values):
        LOG.error("The number of categorical features must match the number of drop values")
        raise ValueError(
            "The number of categorical features must match the number of drop values"
        )

    modified_cat_features = []
    dropped_values = dict(zip(categorical_features, drop_values))
    dropped_columns = []

    for feature in categorical_features:
        dummies = pd.get_dummies(
            df[feature], prefix=feature, prefix_sep="_", drop_first=False, dtype=float
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

    drop_df = pd.DataFrame({"Dropped_Columns": dropped_columns})
    new_columns = drop_df["Dropped_Columns"].tolist()
    dropped_df = pd.DataFrame(columns=new_columns)

    return categorical_encoded, dropped_df


def encode_test_data(
    test_df: pd.DataFrame,
    categorical_features: List[str],
    train_encoded: pd.DataFrame,
    target_column: str,
    weight_column: str,
    weight_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode test data to match the columns of the encoded training data.

    Parameters
    ----------
    test_df : pandas.DataFrame
        Test data to be encoded.
    categorical_features : list of str
        List of column names for categorical variables.
    train_encoded : pandas.DataFrame
        Encoded training data (used to align columns).
    target_column : str
        Name of the target column.
    weight_column : str
        Name of the weight column.
    weight_df : pandas.DataFrame
        Weight column to be added back to the encoded test data.

    Returns
    -------
    test_encoded : pandas.DataFrame
        Encoded test data.
    """

    if target_column in train_encoded.columns:
        train_encoded = train_encoded.drop(columns=[target_column])
    if weight_column in train_encoded.columns:
        train_encoded = train_encoded.drop(columns=[weight_column])

    if target_column in test_df.columns:
        test_df = test_df.drop(columns=[target_column])

    test_encoded = pd.get_dummies(
        test_df, columns=categorical_features, prefix_sep="_", dtype=float
    )

    test_encoded = test_encoded.loc[:, test_encoded.columns.isin(train_encoded.columns)]

    if weight_df is not None:
        test_encoded[weight_column] = weight_df

    return test_encoded


def process_data_pipeline(
    df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    target_column: str,
    weight_column: str,
    sample_size_encode: bool,
    select_encode_values: bool,
    encode_values_to_drop: List[str],
    train_encoded: None | pd.DataFrame,
    test_data: bool,
    numerical_pipeline,
    output_folder: Path,
):
    """
    Pipeline to process input data via encoding and scaling transformations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    numerical_features : list of str
        List of column names for continuous variables.
    categorical_features : list of str
        List of column names for categorical variables.
    target_column : str
        Name of the column to predict.
    weight_column : str
        Optional column name to be used as sample weights.
    sample_size_encode : bool
        If True, use sample size encoding for categorical variables.
    select_encode_values : bool
        If True, use custom value encoding for categorical variables.
    encode_values_to_drop : list of str
        Values to drop for custom encoding.
    train_encoded : pandas.DataFrame or None
        Encoded and scaled training dataset, used to align test data.
    test_data : bool
        If True, the dataframe is test data.
    numerical_pipeline : sklearn.Pipeline or None
        Stored numerical pipeline from training data.
    output_folder : pathlib.Path
        Path to output folder.

    Returns
    -------
    preprocessed_df : pandas.DataFrame
        Scaled and encoded dataframe.
    drop_vals : pandas.DataFrame or None
        Columns removed during the encoding process.
    numerical_pipeline : sklearn.Pipeline or None
        Stored transformation pipeline for continuous variables.
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
            numerical_df = preprocess_numerical_data(
                df=x,
                numerical_features=numerical_features,
                is_test_data=True,
                numerical_pipeline_train=numerical_pipeline,
                output_folder=output_folder,
            )
        else:
            numerical_df, numerical_pipeline = preprocess_numerical_data(
                df=x,
                numerical_features=numerical_features,
                is_test_data=False,
                numerical_pipeline_train=numerical_pipeline,
                output_folder=output_folder,
            )

        if y is not None:
            numerical_df[target_column] = y

        if weight_df is not None:
            numerical_df[weight_column] = weight_df

        return numerical_df, drop_vals, numerical_pipeline

    # just categorical data
    if categorical_features is not None and numerical_features is None:
        if test_data is True:
            test_final = encode_test_data(
                test_df=x,
                categorical_features=categorical_features,
                train_encoded=train_encoded,
                target_column=target_column,
                weight_column=weight_column,
                weight_df=weight_df,
            )
            return test_final, None

        categorical_df, drop_vals = preprocess_categorical_data(
            df=x,
            categorical_features=categorical_features,
            sample_size_encode=sample_size_encode,
            select_encode_values=select_encode_values,
            encode_values_to_drop=encode_values_to_drop,
        )
        if y is not None:
            categorical_df[target_column] = y

        if weight_df is not None:
            categorical_df[weight_column] = weight_df

        return categorical_df, drop_vals, None

    # both categorical and numerical
    if numerical_features and categorical_features is not None:
        x_cat = x.drop(columns=numerical_features)
        if test_data is True:
            numerical_df = preprocess_numerical_data(
                df=x,
                numerical_features=numerical_features,
                is_test_data=True,
                numerical_pipeline_train=numerical_pipeline,
                output_folder=output_folder,
            )

            test_final = encode_test_data(
                test_df=x_cat,
                categorical_features=categorical_features,
                train_encoded=train_encoded,
                target_column=target_column,
                weight_column=weight_column,
                weight_df=weight_df,
            )

            preprocessed_df = pd.concat([numerical_df, test_final], axis=1)

            if weight_df is not None:
                preprocessed_df[weight_column] = weight_df

            return preprocessed_df, None, None

        else:
            numerical_df, numerical_pipeline = preprocess_numerical_data(
                df=x,
                numerical_features=numerical_features,
                is_test_data=False,
                numerical_pipeline_train=numerical_pipeline,
                output_folder=output_folder,
            )

            categorical_df, drop_vals = preprocess_categorical_data(
                df=x_cat,
                categorical_features=categorical_features,
                sample_size_encode=sample_size_encode,
                select_encode_values=select_encode_values,
                encode_values_to_drop=encode_values_to_drop,
            )
            preprocessed_df = pd.concat([numerical_df, categorical_df], axis=1)

    if y is not None:
        preprocessed_df[target_column] = y
        preprocessed_df[target_column] = preprocessed_df[target_column].astype(int)

    if weight_df is not None:
        preprocessed_df[weight_column] = weight_df

    return preprocessed_df, drop_vals, numerical_pipeline

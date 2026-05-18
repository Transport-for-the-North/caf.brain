"""
Created on: 6/11/2024
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import os
from pathlib import Path
from typing import List, Optional

# Third Party
import joblib
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
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
) -> tuple[pd.DataFrame, Optional[Pipeline]]:
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

    Parameters
    ----------
    df: Input dataframe.
    numerical_features: List of string column names that are
                               continuous variables.
    is_test_data: Bool. Used to dictate if test data is used or not.
    numerical_pipeline_train: Always None unless full model flow is
                                     running. Only is relevant to test data.
                                     If None then one of numerical_pipeline.pkl
                                     or scale_csv.csv needs to be available to
                                     apply transformations to train.
    output_folder: Path to output folder.

    Returns
    -------
    Dataframe of only the scaled numerical data set to the input datas index.
    Also, the numerical pipeline information when applicable.
    """

    if is_test_data:
        LOG.info("Processing test data - attempting to apply training transformations")
        numerical_pipeline = None
        method_used = None

        if numerical_pipeline_train is not None:
            try:
                numerical_pipeline = numerical_pipeline_train
                method_used = "in-memory pipeline"
            except ValueError as e:
                LOG.warning("Could not use in-memory pipeline: %s", str(e))

        if numerical_pipeline is None and output_folder is not None:
            numerical_pipeline_path_pkl = os.path.join(output_folder, "numerical_pipeline.pkl")
            try:
                numerical_pipeline = joblib.load(numerical_pipeline_path_pkl)
                method_used = "pickled pipeline"
            except ValueError as e:
                LOG.warning("Could not load pipeline from pickle: %s", str(e))

        if numerical_pipeline is None and output_folder is not None:
            scale_csv_path = os.path.join(output_folder, "scale_csv.csv")
            scale_df = pd.read_csv(scale_csv_path, index_col=0)

            scaler = StandardScaler()
            scaler.mean_ = scale_df.loc["mean"].values
            scaler.scale_ = scale_df.loc["std"].values
            scaler.var_ = scale_df.loc["var"].values

            try:
                numerical_pipeline = Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]
                )
                method_used = "CSV scaler values"
            except ValueError as e:
                LOG.warning("Could not load scaler values from CSV: %s", str(e))

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

        return numerical_df, None

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
    categorical_features: list[str],
    sample_size_encode: bool | None,
    select_encode_values: bool | None,
    encode_values_to_drop: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encodes categorical variables via a choice of methods. Standard
    encoding where the first in each category is dropped is default,
    see pandas.get_dummies documentation. Encoding via sample size or values
    set by the user is also possible.

    Parameters
    ----------
    df: Input dataframe.
    categorical_features: List of string column names that are
                          categorical variables.
    sample_size_encode: Optional bool. If true, the data will be split
                        based on sample size. Variables with the largest
                        sample size will be used as reference class.
    select_encode_values: Optional bool. If True, data is split based
                          on custom values set by the user. Corresponds
                          to encode_values_to_drop.
    encode_values_to_drop: If select_encode_values is True, then this
                           must be a list of strings the length of
                           categorical_features. Position one in the list
                           will link to the first variable provided in
                           categorical_features and so on.

    Returns
    -------
    categorical_df: Dataframe of encoded categorical data set
                    to the input datas index.
    drop_vals: Dataframe of columns removed during the encoding process.
    """
    df.columns = df.columns.astype(str)

    if sample_size_encode:
        categorical_df, drop_vals = sample_size_encode_(
            df=df, categorical_features=categorical_features
        )
        return categorical_df, drop_vals
    if select_encode_values:
        if not encode_values_to_drop:
            raise ValueError("")
        categorical_df, drop_vals = custom_sample_encode(
            df=df, categorical_features=categorical_features, drop_values=encode_values_to_drop
        )
        return categorical_df, drop_vals

    categorical_df = pd.get_dummies(
        df, columns=categorical_features, drop_first=True, dtype=float
    )
    categorical_df.columns = categorical_df.columns.str.replace(".0", "")

    cat_w_all_cols = pd.get_dummies(df, columns=categorical_features, dtype=float)
    cat_w_all_cols.columns = cat_w_all_cols.columns.str.replace(".0", "")

    extra_columns = cat_w_all_cols.columns.difference(categorical_df.columns)
    drop_vals = cat_w_all_cols[extra_columns]

    return categorical_df, drop_vals


def sample_size_encode_(
    df: pd.DataFrame, categorical_features: List[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encodes variables based on sample size. The value that appears most often
    in each of the categorical variables is used as the reference and therefore
    dropped during encoding.

    Parameters
    ----------
    df: Input dataframe.
    categorical_features: List of string column names that are
                          categorical variables.

    Returns
    -------
    Categorical_encoded: Dataframe of encoded categorical data.
    Dropped_df: Dataframe of columns removed during the encoding process.
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
    df: pd.DataFrame, categorical_features: list[str], drop_values: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encodes variables based on user specified values. The values should be
    specified in the order that the variables are listed in categorical_features.
    These will be the values dropped when encoded and therefore used as
    the reference.

    Parameters
    ----------
    df: Input dataframe.
    categorical_features: List of string column names that are
                          categorical variables.
    drop_values: List of strings the length of categorical_features.
                 Position one in the list will link to the first
                 variable provided in categorical_features and so on.

    Returns
    -------
    Categorical_encoded: Dataframe of encoded categorical data.
    Dropped_df: Dataframe of columns removed during the encoding process.
    """
    assert isinstance(categorical_features, list)
    assert isinstance(drop_values, list)
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
    train_encoded: Optional[pd.DataFrame],
    target_column: str | None,
    weight_column: str | None,
    weight_df: pd.Series | None,
) -> pd.DataFrame:
    """
    This encodes test data to match the training data.

    Parameters
    ----------
    test_df: Test data to be encoded.
    categorical_features: List of string column names that are
                          categorical variables.
    train_encoded: Encoded training data that the test data will match.
    target_column: Sting column name of value to predict.
    weight_column: Optional string column value to be used as weight.
    weight_df: The weight column in a series to be added back
               to the encoded test data.

    Returns
    -------
    test_encoded: Encoded test data.
    """
    if train_encoded is None:
        raise ValueError("train_encoded is required")
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

    test_encoded = _align_categorical_dummies(
        test_encoded=test_encoded,
        train_encoded=train_encoded,
        categorical_features=categorical_features,
    )

    if weight_df is not None:
        test_encoded[weight_column] = weight_df

    return test_encoded


def _align_categorical_dummies(
    test_encoded: pd.DataFrame,
    train_encoded: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    """
    Ensures that test_encoded contains all categorical dummy columns
    that appear in train_encoded, adding missing ones with 0.0.

    Numeric columns are ignored and left untouched.

    Parameters
    ----------
    test_encoded:
        Encoded test data that will be aligned to the training data.
    train_encoded:
        Encoded training data that the test data will match.
    categorical_features:
        List of string column names that are categorical variables.
    Returns
    -------
    Aligned test dataframe.
    """

    train_cat_cols = [
        c
        for c in train_encoded.columns
        if any(c.startswith(f"{feat}_") for feat in categorical_features)
    ]

    test_cat_cols = [
        c
        for c in test_encoded.columns
        if any(c.startswith(f"{feat}_") for feat in categorical_features)
    ]
    if set(train_cat_cols) == set(test_cat_cols):
        return test_encoded

    missing = set(train_cat_cols) - set(test_cat_cols)
    for col in missing:
        test_encoded[col] = 0.0

    ordered_cat = test_encoded[train_cat_cols]

    numeric_cols = [c for c in test_encoded.columns if c not in train_cat_cols]
    numeric_df = test_encoded[numeric_cols]

    final = pd.concat([numeric_df, ordered_cat], axis=1)

    return final


def process_data_pipeline(
    df: pd.DataFrame,
    numerical_features: List[str] | None,
    categorical_features: List[str] | None,
    target_column: str | None,
    weight_column: str | None,
    sample_size_encode: bool | None,
    select_encode_values: bool | None,
    encode_values_to_drop: list[str] | None,
    train_encoded: Optional[pd.DataFrame],
    test_data: bool,
    numerical_pipeline,
    output_folder: Path,
    skip_encoding_and_scaling: bool | None = False,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Pipeline]]:
    """
    Pipeline to process input data via encoding and scaling transformations.

    Parameters
    ----------
    df: Input dataframe.
    numerical_features: List of string column names that are
                        continuous variables.
    categorical_features: List of string column names that are
                          categorical variables.
    target_column: String column name of value to predict.
    weight_column: Optional string column value to be used as weight.
    sample_size_encode: Optional bool. If true, the data will be split
                        based on sample size. Variables with the largest
                        sample size will be used as reference class.
    select_encode_values: Optional bool. If True, data is split based
                          on custom values set by the user. Corresponds
                          to encode_values_to_drop.
    encode_values_to_drop: If select_encode_values is True, then this
                           must be a list of strings the length of
                           categorical_features. Position one in the list
                           will link to the first variable provided in
                           categorical_features and so on.
    train_encoded: Either None or an encoded and scaled train dataset.
                   This is to ensure that the corresponding test data
                   is encoding in the same way as the training data.
                   Columns in train and test must match for prediction.
    test_data: Bool, if True then the dataframe passed must be the test data.
    numerical_pipeline: This is the stored numerical pipline used on
                        the training data. Can be left as None if a full
                        model run is not being complete. In that case,
                        output_folder must contain the csv or pkl
                        version of numerical_pipeline.
    output_folder: Path to output folder
    skip_encoding_and_scaling:
        If true, no encoding or scaling applied.

    Returns
    -------
    final_df: Scaled and encoded dataframe.
    drop_vals: Dataframe of columns removed during the encoding process.
    pipeline_out: stored transformation pipeline for continuous variables
    """
    if skip_encoding_and_scaling:
        LOG.info(
            "Skipping encoding and scaling process as " "skip_encoding_and_scaling is True"
        )
        return df, None, None

    if target_column is not None and target_column in df.columns:
        x = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        x = df
        y = None

    final_df = None
    weight_df = None
    drop_vals = None
    pipeline_out = None

    if weight_column is not None and weight_column in df.columns:
        weight_df = df[weight_column]
        x = x.drop(columns=weight_column)

    # just numerical data
    if numerical_features is not None and categorical_features is None:
        if test_data:
            final_df, _ = preprocess_numerical_data(
                df=x,
                numerical_features=numerical_features,
                is_test_data=True,
                numerical_pipeline_train=numerical_pipeline,
                output_folder=output_folder,
            )

            if weight_df is not None:
                final_df[weight_column] = weight_df
        else:
            final_df, pipeline_out = preprocess_numerical_data(
                df=x,
                numerical_features=numerical_features,
                is_test_data=False,
                numerical_pipeline_train=numerical_pipeline,
                output_folder=output_folder,
            )

            if y is not None:
                final_df[target_column] = y

            if weight_df is not None:
                final_df[weight_column] = weight_df

    # just categorical data
    if categorical_features is not None and numerical_features is None:
        if test_data:
            final_df = encode_test_data(
                test_df=x,
                categorical_features=categorical_features,
                train_encoded=train_encoded,
                target_column=target_column,
                weight_column=weight_column,
                weight_df=weight_df,
            )

            if weight_df is not None:
                final_df[weight_column] = weight_df
        else:
            final_df, drop_vals = preprocess_categorical_data(
                df=x,
                categorical_features=categorical_features,
                sample_size_encode=sample_size_encode,
                select_encode_values=select_encode_values,
                encode_values_to_drop=encode_values_to_drop,
            )
            if y is not None:
                final_df[target_column] = y

            if weight_df is not None:
                final_df[weight_column] = weight_df

    # both categorical and numerical
    if numerical_features is not None and categorical_features is not None:
        x_cat = x.drop(columns=numerical_features)
        if test_data:
            numerical_df, _ = preprocess_numerical_data(
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

            final_df = pd.concat([numerical_df, test_final], axis=1)

            if weight_df is not None:
                final_df[weight_column] = weight_df
        else:
            numerical_df, pipeline_out = preprocess_numerical_data(
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
            final_df = pd.concat([numerical_df, categorical_df], axis=1)

            if y is not None:
                final_df[target_column] = y
                if not is_numeric_dtype(final_df[target_column]):
                    raise ValueError(f"Target column '{target_column}' must be numeric")

            if weight_df is not None:
                final_df[weight_column] = weight_df

    if drop_vals is not None and not drop_vals.empty:
        drop_vals.to_csv(os.path.join(output_folder, "dropped_encoding_vals.csv"), index=True)

    if final_df is None:
        raise ValueError("Processing failed: final_df was never created.")

    return final_df, drop_vals, pipeline_out


def _process_data_pipeline_numeric_only(
    df: pd.DataFrame,
    target_column: str,
    numerical_features: list[str],
    output_folder: Path,
    weight_column: str | None = None,
):
    """
    Process only numerical features while preserving raw categorical columns.

    This function is used when the user provides both numerical and categorical
    features but explicitly requests that only numerical features are processed.
    Categorical columns are included in the final dataframe unchanged.

    Parameters
    ----------
    df:
        Pandas Dataframe of your data. Structured or semi-structured tabular
        format.
    target_column:
        Column in your data that is the target variable (Y, dependent variable),
        what you want to predict.
    weight_column:
        Optional string column value to be used as weight.
    numerical_features:
        List of column names (strings) that are continuous variables.
    output_folder:
        Path to output location.

    Returns
    -------
    Pandas dataframe of input data scaled and categorical columns left unchanged.
    """
    if target_column is not None and target_column in df.columns:
        x = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        x = df
        y = None

    weight_df = None
    if weight_column is not None and weight_column in df.columns:
        weight_df = df[weight_column]
        x = x.drop(columns=weight_column)

    x_cat = x.drop(columns=[c for c in numerical_features if c in x.columns])
    numerical_df, _ = preprocess_numerical_data(
        df=x,
        numerical_features=numerical_features,
        is_test_data=False,
        numerical_pipeline_train=None,
        output_folder=output_folder,
    )

    final_df = pd.concat([numerical_df, x_cat], axis=1)

    if y is not None:
        final_df[target_column] = y
        if not is_numeric_dtype(final_df[target_column]):
            raise ValueError(f"Target column '{target_column}' must be numeric")

    if weight_df is not None:
        final_df[weight_column] = weight_df
    return final_df


def _process_data_pipeline_categorical_only(
    df: pd.DataFrame,
    target_column: str,
    categorical_features: list[str],
    weight_column: str | None = None,
):
    """
    Process only categorical features while preserving raw numerical columns.

    This function is used when the user provides both numerical and categorical
    features but explicitly requests that only categorical features are processed.
    Numerical columns are included in the final dataframe unchanged.

    Parameters
    ----------
    df:
        Pandas Dataframe of your data. Structured or semi-structured tabular
        format.
    target_column:
        Column in your data that is the target variable (Y, dependent variable),
        what you want to predict.
    categorical_features:
        List of column names (strings) that are categorical variables.
    weight_column:
        Optional string column value to be used as weight.

    Returns
    -------
    Pandas dataframe of input data encoded and numeric columns left unchanged.
    """
    if target_column is not None and target_column in df.columns:
        x = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        x = df
        y = None

    weight_df = None

    if weight_column is not None and weight_column in df.columns:
        weight_df = df[weight_column]
        x = x.drop(columns=weight_column)

    x_num = x.drop(columns=[c for c in categorical_features if c in x.columns])
    categorical_df, _ = preprocess_categorical_data(
        df=x[categorical_features],
        categorical_features=categorical_features,
        sample_size_encode=None,
        select_encode_values=None,
        encode_values_to_drop=None,
    )
    final_df = pd.concat([x_num, categorical_df], axis=1)

    if y is not None:
        final_df[target_column] = y
        if not is_numeric_dtype(final_df[target_column]):
            raise ValueError(f"Target column '{target_column}' must be numeric")

    if weight_df is not None:
        final_df[weight_column] = weight_df

    return final_df

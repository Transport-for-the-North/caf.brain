"""Created on: 9/9/2025. Original author: Adil Zaheer"""

import logging
import os.path
from pathlib import Path
from typing import Optional
import pandas as pd
from caf.brain.ml import Models
from caf.brain.ml.functions_and_classes.feature_selection.functions import (
    analyse_feature_importance,
)
from caf.brain.ml.functions_and_classes.hparam_optimisation.functions import select_param
from caf.brain.ml.functions_and_classes.model_selection.functions import select_model
from caf.brain.ml.functions_and_classes.process_data_functions.encode_and_scale import (
    process_data_pipeline,
)
from caf.brain.ml.functions_and_classes.process_data_functions.input_data import (
    InitialDataProcessing,
)
from caf.brain.ml.inputs_and_baseclasses.baseclasses import ValidateData

LOG = logging.getLogger(__name__)


def _load_data(data: pd.DataFrame | None, data_path: Path | None) -> pd.DataFrame:
    """
    Loads data into a pandas dataframe.

    Parameters
    ----------
    data: Pandas Dataframe of your data. Structured or semi-structured
          tabular format.
    data_path: Path to your structured or semi-structured tabular data.

    Returns
    -------
    Pandas dataframe.
    """
    if data is not None:
        return data
    if data_path:
        return pd.read_csv(data_path)
    raise ValueError("No data or data_path provided")


def validation(data: pd.DataFrame, custom_index: list[str], target: str) -> bool:
    """
    Validates data against baseclasses to ensure data is suitable for
    further processing.

    Parameters
    ----------
    data: Pandas Dataframe of your data. Structured or semi-structured
          tabular format.
    custom_index: Columns in your data that are to be indexed e.g. year,
                  geography.
    target: Column in your data that is the target variable (Y, dependent
            variable), what you want to predict.

    Returns
    -------
    True if all validation checks pass.

    """
    validator = ValidateData(dataframe=data, custom_index=custom_index, target_column=target)
    validator.index_present()
    validator.target_column_present()
    validator.explanatory_data()
    validator.is_data_numeric()
    validator.data_correct_shape()
    LOG.info("Data is in correct format for caf.brAIn processes")
    return True


def tidy_data(
    classification_prediction: tuple[int, ...] | None,
    output_path: Path,
    categorical_features: list[str],
    numerical_features: list[str],
    custom_index: list[str],
    target: str,
    column_name_to_drop_rows: Optional[list[str]],
    value_in_row: list[str | float | int] | None,
    data_path: Path | None,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Converts semi-structured data into structured inline with machine
    learning standards.

    Parameters
    ----------
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    output_path: Path to output location.
    categorical_features: List of column names (strings) that are
                              categorical variables.
    numerical_features: List of column names (strings) that are
                        continuous variables.
    custom_index: Columns in your data that are to be indexed e.g. year,
                  geography.
    target: Column in your data that is the target variable (Y, dependent
            variable), what you want to predict.
    column_name_to_drop_rows: List of string column names that
                              contain values to drop.
    value_in_row: Corresponding values for column_name_to_drop_rows.
    data: Pandas Dataframe of your data. Structured or semi-structured
          tabular format.
    data_path: Path to your structured or semi-structured tabular data.

    Returns
    -------
    Structured dataframe.

    """
    dataframe = _load_data(data=data, data_path=data_path)
    processor = InitialDataProcessing(
        file_path=None,
        folder_path=None,
        output_path=output_path,
        target_column=target,
        custom_index=custom_index,
        column_name_to_drop_rows=column_name_to_drop_rows,
        value_in_row=value_in_row,
        weight_column=None,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        classification_prediction=classification_prediction,
    )
    processor.df = dataframe
    processed = processor.data_already_split_pipeline(is_test_data=False)
    processed_df = list(processed.values())[0]
    if output_path:
        processed_df.to_csv(os.path.join(output_path, "tidy_data.csv"))
        LOG.info("Tidy data output as output path provided")
    return processed_df


def transform_data(
    data: pd.DataFrame,
    output_path: Path,
    categorical_features: list[str] | None,
    numerical_features: list[str] | None,
    target: str,
    custom_index: list[str],
    sample_size_encode: bool | None = None,
    select_encode_values: bool | None = None,
    encode_values_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    """
    Encode and or scale data where applicable for machine learning modelling.

    Parameters
    ----------
    data: Pandas Dataframe of your data. Structured or semi-structured
          tabular format.
    output_path: Path to output location.
    categorical_features: List of column names (strings) that are
                              categorical variables.
    numerical_features: List of column names (strings) that are
                        continuous variables.
    target: Column in your data that is the target variable (Y, dependent
            variable), what you want to predict.
    custom_index: Columns in your data that are to be indexed e.g. year,
                  geography.
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
    Pandas dataframe of input data encoded and scaled.

    """
    if not validation(data=data, custom_index=custom_index, target=target):
        raise ValueError(
            "Data not suitable for encoding and scaling. Please run \
                          full model flow or tidy_data method prior to \
                          data analysis."
        )
    if not output_path:
        raise ValueError(
            "Please provide an output path to use the \
                         _transform_data"
        )
    # TODO unpacking the variables returned from this is causing linting errors
    preprocessed_df, _, _ = process_data_pipeline(
        df=data,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target,
        weight_column=None,
        sample_size_encode=sample_size_encode,
        select_encode_values=select_encode_values,
        encode_values_to_drop=encode_values_to_drop,
        train_encoded=None,
        test_data=False,
        numerical_pipeline=None,
        output_folder=output_path,
    )

    preprocessed_df.to_csv(os.path.join(output_path, "transformed_data.csv"))
    LOG.info("Transformed data output %s", output_path)
    return preprocessed_df


def feat_selection(
    data: pd.DataFrame,
    output_path: Path,
    target: str,
    weight: str | None,
    custom_index: list[str],
    is_encoded: bool = True,
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
    sample_size_encode: bool | None = None,
    select_encode_values: bool | None = None,
    encode_values_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    """
    Conduct simple feature selection.

    Parameters
    ----------
    is_encoded: is data encoded and scaled.
    data: Pandas Dataframe of your data. Structured or semi-structured
          tabular format.
    output_path: Path to output location.
    target: Column in your data that is the target variable (Y, dependent
            variable), what you want to predict.
    weight: Optional string column value to be used as weight.
    custom_index: Columns in your data that are to be indexed e.g. year,
                  geography.
    categorical_features: List of column names (strings) that are
                              categorical variables.
    numerical_features: List of column names (strings) that are
                        continuous variables.
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
    train_final: feature selected train dataset.
    test_final: feature selected test dataset.
    cols_dropped_by_feat_select: data removed due to feature selection.
    """
    if not validation(data=data, target=target, custom_index=custom_index):
        raise ValueError(
            "Data not suitable for data analysis. Please run \
                          full model flow or tidy_data method prior to \
                          data analysis."
        )

    LOG.warning(
        "Data should already been encoded and scaled where applicable \
                 If this is not the case set is_encoded to false"
    )

    if not is_encoded:
        df = transform_data(
            data=data,
            output_path=output_path,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            target=target,
            custom_index=custom_index,
            sample_size_encode=sample_size_encode,
            select_encode_values=select_encode_values,
            encode_values_to_drop=encode_values_to_drop,
        )
    else:
        df = data

    df_final = analyse_feature_importance(
        train_transformed=df,
        target_column=target,
        weight_column=weight,
        output_path=output_path,
    )
    return df_final


def algorithim_evaluation(
    model_choice: list[Models],
    data: pd.DataFrame,
    output_path: Path,
    target: str,
    weight: str,
    classification_prediction: tuple[int, ...] | None,
    custom_index: list[str],
):
    """
    Evaluate which algorithm is best performing. Algorithms must be from
    the Models enum class.

    Parameters
    ----------
    model_choice: List or one algorithm to use as the base of the model.
                  Available algorithms can be seen in ml_inputs.py or __info__.py.
    data: Pandas Dataframe of your data. Structured or semi-structured
          tabular format.
    output_path: Path to output location.
    target: Column in your data that is the target variable (Y, dependent
            variable), what you want to predict.
    weight: Optional string column value to be used as weight.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    custom_index: Columns in your data that are to be indexed e.g. year,
                  geography.
    Returns
    -------
    Initialised best performing model.
    """
    if not validation(data=data, target=target, custom_index=custom_index):
        raise ValueError(
            "Data not suitable for data analysis. Please run \
                          full model flow or tidy_data method prior to \
                          data analysis."
        )
    if not output_path:
        raise ValueError(
            "Please provide an output path to use \
                         algorithim_evaluation"
        )

    if not isinstance(model_choice, list):
        model = [model_choice]
    else:
        model = model_choice

    selected_model = select_model(
        train=data,
        target_column=target,
        weight_column=weight,
        models_to_test=model,
        classification_prediction=classification_prediction,
        output_folder=output_path,
    )

    return selected_model


def hparam_optim(
    model_choice: Models,
    is_time_series: bool | None,
    data: pd.DataFrame,
    output_path: Path,
    target: str,
    weight: str,
    classification_prediction: tuple[int, ...] | None,
):
    """
    Hyperparamter optimisation for your selected algorithim. Algorithim must
    be part of the Models enum class.

    Parameters
    ----------
    model_choice: List or one algorithm to use as the base of the model.
                Available algorithms can be seen in
                ml_inputs.py or __info__.py.
    is_time_series: If true then data must be time series. Time series
                based characteristics are taken into consideration
                during function execution.
    data: Pandas Dataframe of your data. Structured or semi-structured
          tabular format.
    output_path: Path to output location.
    target: Column in your data that is the target variable (Y, dependent
            variable), what you want to predict.
    weight: Optional string column value to be used as weight.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    Returns
    -------
    Initialised model with the best combination of hyperparameters.

    """
    if not output_path:
        raise ValueError(
            "Please provide an output path to use \
                         hparam_optim"
        )

    if not isinstance(model_choice, list):
        model = [model_choice]
    else:
        model = model_choice

    if len(model) > 1:
        LOG.warning(
            "More than one model selected. The first model will be \
        optimised. To find the best performing model, call algorithim_evaluation or \
                    main_model_selection"
        )

    LOG.warning(
        "Data should be encoded and scaled where applicable. Call \
    _transform_data to do this prior to hyperparameter optimisation"
    )

    selected_model = model[0].get_model()

    final_model = select_param(
        train_final=data,
        target_column=target,
        model_instance=selected_model,
        model_name=model_choice,
        classification_prediction=classification_prediction,
        cv=None,
        weight_column=weight,
        output_folder=output_path,
        is_time_series=is_time_series,
    )

    return final_model

"""Created on: 9/9/2025. Original author: Adil Zaheer"""

# Built-Ins
import logging
import os.path
from pathlib import Path
from typing import Optional

# Third Party
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, ElasticNet

# Local Imports
from caf.brain.ml._functions._ml_inputs import Models
from caf.brain.ml._functions.data_analysis.functions import (
    pre_forecast_data_analysis,
    pre_forecast_data_analysis_classification,
)
from caf.brain.ml._functions.feature_selection.functions import (
    analyse_feature_importance,
)
from caf.brain.ml._functions.hparam_optimisation.functions import (
    select_param,
)
from caf.brain.ml._functions.model_selection.functions import select_model, initialise_model
from caf.brain.ml._functions.process_data_functions.encode_and_scale import (
    process_data_pipeline,
)
from caf.brain.ml._functions.process_data_functions.input_data import (
    InitialDataProcessing,
)
from caf.brain.ml._functions._baseclasses import ValidateData
from caf.brain.ml._functions.process_data_functions.split_data_into_ttv import (
    simple_train_test_split,
    stratified_split_with_categories,
)

LOG = logging.getLogger(__name__)


def _load_data(data: Optional[pd.DataFrame], data_path: Path | None) -> pd.DataFrame:
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


def validation(data: pd.DataFrame, custom_index: list[str] | None, target: str) -> bool:
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
    data_path: Path | None,
    classification_prediction: tuple[int, ...] | None,
    output_path: Path | str,
    categorical_features: list[str],
    numerical_features: list[str],
    target: str,
    custom_index: list[str] | None = None,
    weight: str | None = None,
    column_name_to_drop_rows: Optional[list[str]] = None,
    value_in_row: list[str | float | int] | None = None,
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Converts semi-structured data into structured inline with machine
    learning standards.

    Parameters
    ----------
    data_path: Path to your structured or semi-structured tabular data.
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
    weight: Optional string column value to be used as weight.
    column_name_to_drop_rows: List of string column names that
                              contain values to drop.
    value_in_row: Corresponding values for column_name_to_drop_rows.
    data: Pandas Dataframe of your data. Structured or semi-structured
          tabular format.

    Returns
    -------
    Structured dataframe.

    """
    output_path = Path(output_path)

    dataframe = _load_data(data=data, data_path=data_path)
    processor = InitialDataProcessing(
        file_path=None,
        folder_path=None,
        output_path=output_path,
        target_column=target,
        custom_index=custom_index,
        column_name_to_drop_rows=column_name_to_drop_rows,
        value_in_row=value_in_row,
        weight_column=weight,
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
    output_path: Path | str,
    categorical_features: list[str] | None,
    numerical_features: list[str] | None,
    target: str,
    custom_index: list[str] | None = None,
    weight: str | None = None,
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
    weight: Optional string column value to be used as weight.
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
    output_path = Path(output_path)

    if not validation(data=data, custom_index=custom_index, target=target):
        raise ValueError(
            "Data not suitable for encoding and scaling. Please run \n"
            "full model flow or tidy_data method prior to \n"
            "data analysis."
        )
    if not output_path:
        raise ValueError("Please provide an output path to use the _transform_data")

    preprocessed_df, _, _ = process_data_pipeline(
        df=data,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target,
        weight_column=weight,
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
    output_path: Path | str,
    categorical_features: list[str] | None,
    numerical_features: list[str] | None,
    target: str,
    custom_index: list[str] | None = None,
    is_encoded: bool = True,
    weight: str | None = None,
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
    output_path = Path(output_path)

    if not validation(data=data, target=target, custom_index=custom_index):
        raise ValueError(
            "Data not suitable for data analysis. Please run \n"
            "full model flow or tidy_data method prior to \n"
            "data analysis."
        )

    LOG.warning(
        "Data should already been encoded and scaled where applicable \n"
        "If this is not the case set is_encoded to false"
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


def algorithm_evaluation(
    model_choice: list[Models] | Models,
    data: pd.DataFrame,
    output_path: Path | str,
    target: str,
    custom_index: list[str] | None = None,
    weight: str | None = None,
    classification_prediction: tuple[int, ...] | None = None,
) -> BaseEstimator:
    """
    Evaluate which algorithm is best performing. Algorithms must be from
    the Models enum class. It is advised to ensure data is in an optimal
    state in order to get accurate results. This means data is encoded
    and scaled where applicable and feature selection is applied. This can be
    done with:
        from caf.brain.ml import feat_selection, transform_data, algorithm_evaluation

    Parameters
    ----------
    model_choice: List or one algorithm to use as the base of the model.
                  Available algorithms can be seen in _ml_inputs.py or __info__.py.
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
    output_path = Path(output_path)

    LOG.warning(
        "It is advised to run both transform_data and feat_selection \n"
        "prior to algorithm_evaluation."
    )

    if not validation(data=data, target=target, custom_index=custom_index):
        raise ValueError(
            "Data not suitable for algorithm evaluation. Please run \n"
            "full model flow or tidy_data method prior to \n"
            "algorithm evaluation."
        )
    if not output_path:
        raise ValueError("Please provide an output path to use algorithm_evaluation")

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
    data: pd.DataFrame,
    output_path: Path | str,
    target: str,
    is_time_series: bool | None = None,
    weight: str | None = None,
    classification_prediction: tuple[int, ...] | None = None,
) -> BaseEstimator:
    """
    Hyperparameter optimisation for your selected algorithm. Algorithm must
    be part of the Models enum class.

    Parameters
    ----------
    model_choice: List or one algorithm to use as the base of the model.
                Available algorithms can be seen in
                _ml_inputs.py or __info__.py.
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
        raise ValueError("Please provide an output path to use hparam_optim")

    output_path = Path(output_path)

    if not isinstance(model_choice, list):
        model = [model_choice]
    else:
        model = model_choice

    if len(model) > 1:
        LOG.warning(
            "More than one model selected. The first model will be \n"
            "optimised. To find the best performing model, call algorithm_evaluation or \n"
            "main_model_selection"
        )

    LOG.warning(
        "Data should be encoded and scaled where applicable. Call \n"
        "_transform_data to do this prior to hyperparameter optimisation"
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


def evaluate_data(
    data: pd.DataFrame,
    output_path: Path | str,
    categorical_features: list[str] | None,
    numerical_features: list[str] | None,
    target: str,
    classification_prediction: tuple[int, ...] | None,
    weight: str | None = None,
    allow_transformations: bool = True,
    is_time_series: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Data analysis for structured tabular numerical data. Where applicable,
    the following tests are performed:
        - Multicolinearity (VIF)
        - Heteroscedasticity (Breusch-Pagan & White Test)
        - Linearity (Correlation coefficient)
        - Normality (Shapiro-Wilk)
        - Autocorrelation (Durbin-Watson)

    Data transformations (log and scaling) are applied to numerical features
    only if permitted and required.

    Parameters
    ----------
    data: Pandas Dataframe of your data. Structured or semi-structured
          tabular format.
    output_path: Path to output location.
    target: Column in your data that is the target variable (Y, dependent
            variable), what you want to predict.
    weight: Optional string column value to be used as weight.
    categorical_features: List of column names (strings) that are
                          categorical variables.
    numerical_features: List of column names (strings) that are
                        continuous variables.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    allow_transformations: Whether to apply transformations.
    is_time_series: If true then data must be time series. Time series
                    based characteristics are taken into consideration
                    during function execution.
    Returns
    -------
    train_transformed: training data post transformation.
    test_transformed: test data post transformation.
    """

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if not validation(data=data, target=target, custom_index=None):
        raise ValueError(
            "Data not suitable for data analysis. Please run \n"
            "full model flow or tidy_data method prior to \n"
            "data analysis."
        )

    LOG.info(
        "Starting data evaluation for %s prediction",
        "classification" if classification_prediction else "regression",
    )

    # split unscaled data
    train_unscaled, test_unscaled, validate = stratified_split_with_categories(
        df=data,
        categorical_features=categorical_features,
        target_column=target,
        weight_column=weight,
        split_size=None,
        validation_path=None,
        index_columns=None,
        output_path=output_path,
    )

    # encode / scale train
    train_scaled, _, pipeline_out = process_data_pipeline(
        df=train_unscaled,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target,
        weight_column=weight,
        sample_size_encode=True,
        select_encode_values=None,
        encode_values_to_drop=None,
        train_encoded=None,
        test_data=False,
        numerical_pipeline=None,
        output_folder=output_path,
    )

    # encode / scale test
    test_scaled, _, _ = process_data_pipeline(
        df=test_unscaled,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target,
        weight_column=weight,
        sample_size_encode=True,
        select_encode_values=None,
        encode_values_to_drop=None,
        train_encoded=train_scaled,
        test_data=True,
        numerical_pipeline=pipeline_out,
        output_folder=output_path,
    )

    # train/test split for fitting (scaled data)
    x_train, x_test, y_train, y_test, x_train_weight = simple_train_test_split(
        df=train_scaled,
        target_column=target,
        weight_column=weight,
    )

    # initialise model for diagnostics
    if classification_prediction:
        model_initialised = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model_initialised = ElasticNet(random_state=42)

    x_train_model_fit, residuals, mse = initialise_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_train_weight=x_train_weight,
        output_folder=output_path,
        model_initialised=model_initialised,
        classification_prediction=classification_prediction,
    )

    # run data analysis
    if classification_prediction:
        train_transformed, test_transformed = pre_forecast_data_analysis_classification(
            train_scaled=train_scaled,
            test_scaled=test_scaled,
            train_unscaled=train_unscaled,
            target=target,
            numerical_features=numerical_features,
            output_path=output_path,
            classification_prediction=classification_prediction,
        )
    else:
        train_transformed, test_transformed = pre_forecast_data_analysis(
            output_folder=output_path,
            residuals=residuals,
            model_fit=x_train_model_fit,
            model_initialised=model_initialised,
            x_test=x_test,
            y_test=y_test,
            train_scaled=train_scaled,
            test_scaled=test_scaled,
            train_unscaled=train_unscaled,
            test_unscaled=test_unscaled,
            numerical_pipeline=pipeline_out,
            target_column=target,
            weight_column=weight,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            is_time_series=is_time_series,
            allow_transformations=allow_transformations,
        )

    LOG.info("Data evaluation complete. Results saved to %s", output_path)
    LOG.info("Check 'data_issues_present.csv' for detected issues")

    return train_transformed, test_transformed

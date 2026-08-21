"""caf.brAIn Machine Learning API"""

# Built-Ins
import logging
import os.path
import warnings
from pathlib import Path
from typing import Optional, Any

# Third Party
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Local Imports
from caf.brain.ml._functions._baseclasses import ValidateData
from caf.brain.ml._functions._ml_inputs import (
    Models,
    XGBClassifierBinary,
    XGBClassifierMulticlass,
)
from caf.brain.ml._functions.data_analysis.functions import (
    pre_forecast_data_analysis,
)
from caf.brain.ml._functions.feature_selection.functions import (
    analyse_feature_importance,
    combine_results,
)
from caf.brain.ml._functions.hparam_optimisation.functions import (
    select_param,
)
from caf.brain.ml._functions.model_selection.functions import (
    calculate_final_coefficients,
    initialise_model,
    select_model,
)
from caf.brain.ml._functions.process_data_functions.encode_and_scale import (
    _process_data_pipeline_categorical_only,
    _process_data_pipeline_numeric_only,
    process_data_pipeline,
)
from caf.brain.ml._functions.process_data_functions.input_data import (
    InitialDataProcessing,
)
from caf.brain.ml._functions.process_data_functions.split_data_into_ttv import (
    simple_train_test_split,
    split_by_column_value,
    stratified_split_with_categories,
)

LOG = logging.getLogger(__name__)


def _load_data(data: Optional[pd.DataFrame], data_path: Path | None) -> pd.DataFrame:
    """
    Loads data into a pandas dataframe.

    Parameters
    ----------
    data
        Pandas Dataframe of your data. Structured or semi-structured tabular
        format.
    data_path
        Path to your structured or semi-structured tabular data.

    Returns
    -------
    Pandas dataframe.
    """
    if data is not None:
        return data
    if data_path:
        return pd.read_csv(data_path)
    raise ValueError("No data or data_path provided")


def tidy_data(
    data_path: Path | None,
    classification_prediction: tuple[int, ...] | None,
    output_path: Path | str,
    categorical_features: list[str] | None,
    numerical_features: list[str] | None,
    target: str,
    custom_index: list[str] | None = None,
    weight: str | None = None,
    column_name_to_drop_rows: Optional[list[str]] = None,
    value_in_row: list[str | float | int] | None = None,
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Converts semi-structured data into structured for machine learning.

    Parameters
    ----------
    data_path
        Path to your structured or semi-structured tabular data.
    classification_prediction
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.
    output_path
        Path to output location.
    categorical_features
        List of column names (strings) that are categorical variables.
    numerical_features
        List of column names (strings) that are continuous variables.
    custom_index
        Columns in your data that are to be indexed e.g. year, geography.
    target
        Column in your data that is the target variable (Y, dependent variable),
        what you want to predict.
    weight
        Optional string column value to be used as weight.
    column_name_to_drop_rows
        List of string column names that contain values to drop.
    value_in_row
        Corresponding values for column_name_to_drop_rows.
    data
        Pandas Dataframe of your data. Structured or semi-structured tabular
        format.

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
        model_choice=None,
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
    process_numeric_only: bool = False,
    process_categorical_only: bool = False,
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
    data
        Pandas Dataframe of your data. Structured or semi-structured tabular
        format.
    output_path
        Path to output location.
    categorical_features
        List of column names (strings) that are categorical variables.
    numerical_features
        List of column names (strings) that are continuous variables.
    target
        Column in your data that is the target variable (Y, dependent variable),
        what you want to predict.
    process_numeric_only:
        Only use if you have both continuous and categorical data. If true,
        only numeric data is transformed (scaled) and categorical data is left
        unchanged.
    process_categorical_only:
        Only use if you have both continuous and categorical data. If true,
        only categorical data is transformed (encoded) and numerical data is
        left unchanged.
    custom_index
        Columns in your data that are to be indexed e.g. year, geography.
    weight
        Optional string column value to be used as weight.
    sample_size_encode
        Optional bool. If true, the data will be split based on sample size.
        Variables with the largest sample size will be used as reference class.
    select_encode_values
        Optional bool. If True, data is split based on custom values set by the
        user. Corresponds to encode_values_to_drop.
    encode_values_to_drop
        If select_encode_values is True, then this must be a list of strings
        the length of categorical_features. Position one in the list will link
        to the first variable provided in categorical_features and so on.

    Returns
    -------
    Pandas dataframe of input data encoded and scaled.

    """
    output_path = Path(output_path)

    if not ValidateData(
        dataframe=data, custom_index=custom_index, target_column=target
    ).validate():
        raise ValueError(
            "Data not suitable for encoding and scaling. Please run \n"
            "full model flow or tidy_data method prior to \n"
            "data analysis."
        )
    if not output_path:
        raise ValueError("Please provide an output path to use the _transform_data")

    if process_numeric_only:
        if numerical_features is None:
            raise ValueError(
                "Numerical features are required if processing only numerical features"
            )
        preprocessed_df = _process_data_pipeline_numeric_only(
            df=data,
            target_column=target,
            numerical_features=numerical_features,
            output_folder=output_path,
            weight_column=weight,
        )
        preprocessed_df.to_csv(os.path.join(output_path, "transformed_data.csv"))
        LOG.info("Transformed data output %s", output_path)
        return preprocessed_df

    if process_categorical_only:
        if categorical_features is None:
            raise ValueError(
                "Categorical features are required if processing only categorical features"
            )
        preprocessed_df = _process_data_pipeline_categorical_only(
            df=data,
            target_column=target,
            categorical_features=categorical_features,
            weight_column=weight,
        )
        preprocessed_df.to_csv(os.path.join(output_path, "transformed_data.csv"))
        LOG.info("Transformed data output %s", output_path)
        return preprocessed_df

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


def feature_selection(
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
    test_data: pd.DataFrame | None = None,
) -> tuple[DataFrame, DataFrame] | tuple[DataFrame, None]:
    """
    Conduct simple feature selection.

    Parameters
    ----------
    is_encoded
        is data encoded and scaled.
    data
        Pandas Dataframe of your data. Structured or semi-structured tabular
        format.
    output_path
        Path to output location.
    target
        Column in your data that is the target variable (Y, dependent variable),
        what you want to predict.
    weight
        Optional string column value to be used as weight.
    custom_index
        Columns in your data that are to be indexed e.g. year, geography.
    categorical_features
        List of column names (strings) that are categorical variables.
    numerical_features
        List of column names (strings) that are continuous variables.
    sample_size_encode
        Optional bool. If true, the data will be split based on sample size.
        Variables with the largest sample size will be used as reference class.
    select_encode_values
        Optional bool. If True, data is split based on custom values set by
        the user. Corresponds to encode_values_to_drop.
    encode_values_to_drop
        If select_encode_values is True, then this must be a list of strings
        the length of categorical_features. Position one in the list will link
        to the first variable provided in categorical_features and so on.
    test_data:
        Optional test data. Only provide if you want to apply the feature selection
        results to test data. This would imply that the data provided to the
        data argument is your training data.
    Returns
    -------
    Feature selected dataset and feature selected test dataset if test_data
    provided.
    """
    output_path = Path(output_path)

    if not ValidateData(
        dataframe=data, custom_index=custom_index, target_column=target
    ).validate():
        raise ValueError(
            "Data not suitable for data analysis. Please run "
            "full model flow or tidy_data method prior to "
            "data analysis."
        )

    LOG.warning(
        "Data should already been encoded and scaled where applicable "
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

    if test_data is not None:
        warnings.warn(
            "You've provided test data meaning feature selection results "
            "are being applied to test_data."
        )
        test_final, _ = combine_results(
            train_final=df_final,
            target_column=target,
            weight_column=weight,
            test=test_data,
        )
        return df_final, test_final

    return df_final, None


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
    Evaluate which algorithm is best performing.

    Algorithms must be from the Models enum class. It is advised to ensure data
    is in an optimal state in order to get accurate results. This means data is
    encoded and scaled where applicable and feature selection is applied. This
    can be done with:

    ``from caf.brain.ml import feat_selection, transform_data, algorithm_evaluation``

    Parameters
    ----------
    model_choice
        List or one algorithm to use as the base of the model.
        Available algorithms can be seen in _ml_inputs.py or __info__.py.
    data
        Pandas Dataframe of your data. Structured or semi-structured
        tabular format.
    output_path
        Path to output location.
    target
        Column in your data that is the target variable (Y, dependent
        variable), what you want to predict.
    weight
        Optional string column value to be used as weight.
    classification_prediction
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.
    custom_index
        Columns in your data that are to be indexed e.g. year, geography etc.
    Returns
    -------
    Initialised best performing model.
    """
    output_path = Path(output_path)

    warnings.warn(
        "It is advised to run both transform_data and feat_selection \n"
        "prior to algorithm_evaluation."
    )

    if not ValidateData(
        dataframe=data, custom_index=custom_index, target_column=target
    ).validate():
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

    xgb_selected = any(
        m in (Models.XGBOOST_CLASSIFIER, Models.XGBOOST_MULTICLASS) for m in model
    )

    if xgb_selected:
        data = InitialDataProcessing.xgboost_preparation(
            df=data,
            target_column=target,
            classification_prediction=classification_prediction,
            model_choice=model_choice,
        )

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
    Hyperparameter optimisation for your selected algorithm.

    Algorithm must be part of the Models enum class.

    Parameters
    ----------
    model_choice
        List or one algorithm to use as the base of the model.
        Available algorithms can be seen in _ml_inputs.py or __info__.py.
    is_time_series
        If true then data must be time series. Time series based
        characteristics are taken into consideration during function execution.
    data
        Pandas Dataframe of your data. Structured or semi-structured tabular
        format.
    output_path
        Path to output location.
    target
        Column in your data that is the target variable (Y, dependent variable),
        what you want to predict.
    weight
        Optional string column value to be used as weight.
    classification_prediction
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.
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
        warnings.warn(
            "More than one model selected. The first model will be \n"
            "optimised. To find the best performing model, call algorithm_evaluation or \n"
            "main_model_selection"
        )

    warnings.warn(
        "Data should be encoded and scaled where applicable. Call \n"
        "_transform_data to do this prior to hyperparameter optimisation"
    )

    selected_model = model[0].get_model()
    selected_enum = model[0]
    if selected_enum in (Models.XGBOOST_CLASSIFIER, Models.XGBOOST_MULTICLASS):
        data = InitialDataProcessing.xgboost_preparation(
            df=data,
            target_column=target,
            classification_prediction=classification_prediction,
            model_choice=model_choice,
        )

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
    classification_prediction: tuple[int, ...] | None = None,
    weight: str | None = None,
    is_time_series: bool = False,
) -> None:
    """
    Data analysis for structured tabular data.

    Tests conducted depend on if the problem is classification or regression
    and if the data is categorical, numerical or both.

    Parameters
    ----------
    data
        Pandas Dataframe of your data. Structured or semi-structured
        tabular format. This data should not yet be scaled or encoded.
    output_path
        Path to output location.
    target
        Column in your data that is the target variable (Y, dependent
        variable), what you want to predict.
    weight
        Optional string column value to be used as weight.
    categorical_features
        List of column names (strings) that are categorical variables.
    numerical_features
        List of column names (strings) that are continuous variables.
    classification_prediction
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem. Must be provided if the problem is
        classification.
    is_time_series
        If true then data must be time series. Time series based characteristics
        are taken into consideration during function execution.
    Returns
    -------
    None
    """

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if not ValidateData(dataframe=data, custom_index=None, target_column=target).validate():
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
    train_unscaled, test_unscaled, _ = simple_data_split(
        data=data, target=target, weight=weight, output_path=output_path
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

    x_train_model_fit, residuals, _ = initialise_model(
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
    _ = pre_forecast_data_analysis(
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
        allow_transformations=False,
    )

    LOG.info("Data evaluation complete. Results saved to %s", output_path)
    LOG.info("Check 'data_issues_present.csv' for detected issues")


def simple_data_split(
    data: pd.DataFrame,
    target: str,
    output_path: Path | str,
    classification_prediction: tuple[int, ...] | None = None,
    split_by_value: str | None = None,
    custom_index: list[str] | None = None,
    weight: str | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Function to split data into training, test and validation (if applicable).

    If split_by_value is provided then data is split by a specific index value.
    The value must correspond to the index column used. The index column must only
    be one column specified (no multi-index). For example, index_columns: year,
    split_by_value: '2019' means everything pre-2019 is training and everything
    post-2019 is test.

    If split_by_value is not provided then data is split into train, test, and
    validate sets using a simple train test split. If you have rare values, then
    there is a chance that some categories aren't represented in both train and
    test. For complex data like this, it is best to use caf.brains full model
    flow as stratified split is availble which mitigates this issue.

    Parameters
    ----------
    data:
        Pandas Dataframe of your data. Structured or semi-structured
        tabular format. This data should not yet be scaled or encoded.
    output_path:
        Path to output location.
    target:
        Column in your data that is the target variable (Y, dependent
        variable), what you want to predict.
    weight:
        Optional string column value to be used as weight.
    categorical_features:
        List of column names (strings) that are categorical variables.
    custom_index:
        Columns in your data that are to be indexed e.g. year, geography etc.
    split_by_value:
        Optional string that links to custom_index. The value in the index
        column to split the data into training and test.
    classification_prediction:
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.
    Returns
    -------
    Train, test and validate dataframes.
    """

    if split_by_value is not None:
        if custom_index is None:
            raise ValueError(
                "If split_by_value is set, then custom_index must"
                "not be None. split_by_value must correspond to a"
                "value inside the index column provided."
            )
        train, test, validate = split_by_column_value(
            df=data,
            index_columns=custom_index,
            split_by_value=split_by_value,
            weight_column=weight,
            target_column=target,
            validation_path=None,
            output_path=output_path,
        )

        return train, test, validate

    if classification_prediction is not None:
        train, test, validate = stratified_split_with_categories(
            df=data,
            categorical_features=categorical_features,
            target_column=target,
            weight_column=weight,
            split_size=None,
            validation_path=None,
            index_columns=custom_index,
            output_path=output_path,
        )
        return train, test, validate

    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
    )
    validate = None
    if weight in test.columns:
        test = test.drop(columns=weight)

    if target in test.columns:
        validate = pd.DataFrame(
            {target: test[target]},
            index=test.index,
        )
        validate.to_csv(os.path.join(output_path, "validate.csv"), index=True)
        test = test.drop(columns=target)

    train.to_csv(os.path.join(output_path, "train.csv"), index=True)
    test.to_csv(os.path.join(output_path, "test.csv"), index=True)

    return train, test, validate


def simple_prediction(
    model: Any,
    test: pd.DataFrame,
    target_column: str,
    output_folder: Path,
    validation: pd.DataFrame | None = None,
    weight_column: str | None = None,
    classification_prediction: tuple[int, ...] | None = None,
) -> None:
    """
    Generate predictions and final model coefficients, saving results to disk.

    Parameters
    ----------
    model:
        Fitted final model for prediction on unseen (test) data.
    test:
        Dataframe of final test data post feature selection.
    target_column:
        String column name of value to predict.
    output_folder:
        Path to output location.
    validation:
        Validation data if available.
    weight_column:
        Optional string column value to be used as weight.
    classification_prediction:
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.

    Returns
    -------
    predictions: Predicted values based on the test data and set to the same
                 index.
    """
    mse = None
    if validation is not None and not target_column:
        raise ValueError(
            "Please provide a target column for prediction as you "
            "have passed a validation set of data. The target column "
            "if a string of the column title."
        )

    if target_column in test.columns:
        test = test.drop(columns=target_column)

    if weight_column in test.columns:
        weight = test[weight_column].to_numpy().flatten()
    else:
        weight = None

    if classification_prediction is not None:
        if validation is not None:
            validation = validation.loc[test.index]
            if isinstance(model, LinearSVC):
                pred_classes = model.predict(test)
                y_true = validation[target_column].values
                accuracy = accuracy_score(y_true, pred_classes, sample_weight=weight)
            else:
                pred_probs = model.predict_proba(test)
                if isinstance(model, XGBClassifierMulticlass):
                    pred_classes = np.argmax(pred_probs, axis=1)
                else:
                    pred_classes = model.classes_[np.argmax(pred_probs, axis=1)]
                y_true = validation[target_column].values
                accuracy = accuracy_score(y_true, pred_classes, sample_weight=weight)

            LOG.info("Accuracy: %s", accuracy)
            accuracy_df = pd.DataFrame({"accuracy": [accuracy]})
            accuracy_df.to_csv(os.path.join(output_folder, "model_performance.csv"))
        else:
            if isinstance(model, LinearSVC):
                pred_classes = model.predict(test)
            else:
                pred_probs = model.predict_proba(test)
                if isinstance(model, XGBClassifierMulticlass):
                    pred_classes = np.argmax(pred_probs, axis=1)
                else:
                    pred_classes = model.classes_[np.argmax(pred_probs, axis=1)]
        if isinstance(model, (XGBClassifierBinary, XGBClassifierMulticlass)):
            mapping = dict(enumerate(classification_prediction))
            pred_classes = pd.Series(pred_classes).map(mapping).to_numpy()
        predictions = pred_classes

    else:
        predictions = model.predict(test)
        if validation is not None:
            validation = validation.loc[test.index]
            r2 = r2_score(validation[target_column], predictions, sample_weight=weight)
            mse = mean_squared_error(
                validation[target_column], predictions, sample_weight=weight
            )
            LOG.info("r2: %s", r2)
            LOG.info("mse: %s", mse)
            metrics_df = pd.DataFrame({"r2": [r2], "mse": [mse]})
            metrics_df.to_csv(os.path.join(output_folder, "model_performance.csv"))

    coeff_df = calculate_final_coefficients(
        model=model,
        test_data=test,
        training_mse=mse,
        predictions=predictions,
        validation_data=validation,
        target_column=target_column,
        is_classification=classification_prediction,
        drop_vals=None,
        cols_dropped_by_feat_select=None,
    )
    if coeff_df is not None:
        coeff_df.to_csv(
            os.path.join(output_folder, "final_model_coefficients.csv"), index=False
        )

    final_predictions = pd.DataFrame(
        {"predicted_target_column": predictions}, index=test.index
    )
    final_predictions.to_csv(os.path.join(output_folder, "final_predictions.csv"))


def visualise_model_performance(
    model: Any,
    test: str | pd.DataFrame,
    target: str,
    output_folder: Path | str,
    validation: str | pd.DataFrame,
    is_classification: bool = False,
    index_columns: list[str] | None = None,
    weight_column: str | None = None,
) -> None:
    """
    Visualise model performance for regression or classification.

    All outputs are saved to output_folder.

    Parameters
    ----------
    model:
        Path to fitted final model for prediction on unseen (test) data.
    test:
        Dataframe of final test data that was used for prediction.
    target:
        String column name of value to predict.
    output_folder:
        Path to output location.
    index_columns:
        Columns in your data that are to be indexed e.g. year, geography etc.
    validation:
        Validation data that aligns with the final test data as truth values.
    weight_column:
        Optional string column value to be used as weight.
    is_classification:
        If true, classification visualisation will be done.

    Returns
    -------
    None
    """
    if isinstance(output_folder, str):
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if isinstance(model, str):
        LOG.info("Loading model from %s", model)
        model = joblib.load(model)
    LOG.info("Model already loaded and being used")

    if isinstance(test, str):
        LOG.info("Loading test data from %s", test)
        test = pd.read_csv(test)
    LOG.info("Test data loaded and being used")

    if isinstance(validation, str):
        LOG.info("Loading validation data from %s", validation)
        validation = pd.read_csv(validation)
    LOG.info("validation data loaded and being used")

    if index_columns:
        test = test.set_index(index_columns)
        validation = validation.set_index(index_columns)

    weight = test[weight_column] if weight_column else None
    y_pred = model.predict(test)
    val = validation[target]
    metrics = {}

    if not is_classification:
        metrics["r2"] = r2_score(val, y_pred, sample_weight=weight)
        metrics["mse"] = mean_squared_error(val, y_pred, sample_weight=weight)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(val, y_pred, sample_weight=weight)
        pd.DataFrame([metrics]).to_csv(output_folder / "metrics.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(val, y_pred, alpha=0.5)
        lims = (float(min(val.min(), y_pred.min())), float(max(val.max(), y_pred.max())))
        ax.plot(lims, lims, "r--")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual (Test)")
        fig.savefig(output_folder / "pred_vs_actual_test.png", dpi=300)
        plt.close(fig)

        residuals = val - y_pred
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Predicted (Test)")
        fig.savefig(output_folder / "residuals_test.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title("Residual Distribution (Test)")
        fig.savefig(output_folder / "residual_distribution_test.png", dpi=300)
        plt.close(fig)

        return

    metrics["accuracy_test"] = accuracy_score(val, y_pred, sample_weight=weight)
    metrics["f1_test"] = f1_score(val, y_pred, average="weighted", sample_weight=weight)
    metrics["classification_report_test"] = classification_report(val, y_pred)
    pd.DataFrame([metrics]).to_csv(output_folder / "metrics.csv", index=False)

    cm = confusion_matrix(val, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test)")
    fig.savefig(output_folder / "confusion_matrix_test.png", dpi=300)
    plt.close(fig)

    if hasattr(model, "predict_proba"):
        classes = model.classes_

        if len(classes) == 2:
            y_proba_test = model.predict_proba(test)
            positive_class = classes[-1]
            positive_index = list(classes).index(positive_class)

            # ROC
            fpr, tpr, _ = roc_curve(
                val, y_proba_test[:, positive_index], pos_label=positive_class
            )
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_title("ROC Curve (Test)")
            ax.legend()
            fig.savefig(output_folder / "roc_curve_test.png", dpi=300)
            plt.close(fig)

            # PR
            precision, recall, _ = precision_recall_curve(
                val, y_proba_test[:, positive_index], pos_label=positive_class
            )
            pr_auc = auc(recall, precision)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
            ax.set_title("Precision-Recall Curve (Test)")
            ax.legend()
            fig.savefig(output_folder / "precision_recall_test.png", dpi=300)
            plt.close(fig)
    return


def dependant_variable_testing(
    data: str | pd.DataFrame,
    target: str,
    output_folder: Path | str,
    weight_column: str | None = None,
) -> None:
    """
    Dependant variable testing for pre-modelling analysis.

    Parameters
    ----------
    data:
        Dataframe or CSV path containing the dataset.
    target:
        Column name of the dependent variable.
    output_folder:
        Folder where outputs will be saved.
    weight_column:
        Optional weight column.

    Returns
    -------
    None
    """
    if isinstance(output_folder, str):
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if isinstance(data, str):
        data = pd.read_csv(data)

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    y = data[target]
    w = data[weight_column] if weight_column else None

    unique_vals = y.nunique(dropna=True)
    if unique_vals <= 10:
        var_type = "categorical"
    else:
        var_type = "continuous"

    if var_type == "continuous":
        stats = {
            "count": y.count(),
            "mean": y.mean(),
            "std": y.std(),
            "min": y.min(),
            "25%": y.quantile(0.25),
            "50% (median)": y.median(),
            "75%": y.quantile(0.75),
            "max": y.max(),
            "skew": y.skew(),
            "kurtosis": y.kurtosis(),
            "unique_values": unique_vals,
        }

        pd.DataFrame([stats]).to_csv(output_folder / "continuous_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.histplot(y, kde=True, ax=ax)
        ax.set_title(f"Distribution of {target}")
        fig.savefig(output_folder / f"{target}_hist.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=y, ax=ax)
        ax.set_title(f"Boxplot of {target}")
        fig.savefig(output_folder / f"{target}_boxplot.png", dpi=300)
        plt.close(fig)

        if w is not None:
            weighted_mean = np.average(y, weights=w)
            pd.DataFrame([{"weighted_mean": weighted_mean}]).to_csv(
                output_folder / "weighted_stats.csv", index=False
            )

        return

    counts = y.value_counts(dropna=False)
    counts.to_csv(output_folder / "categorical_counts.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(y=y, ax=ax)
    ax.set_title(f"Category Counts for {target}")
    plt.xticks(rotation=45)
    fig.savefig(output_folder / f"{target}_countplot.png", dpi=300)
    plt.close(fig)

    if weight_column is not None:
        col = weight_column
        weighted_counts = data.groupby(target)[col].sum()
        weighted_counts.to_csv(output_folder / "categorical_weighted_counts.csv")

    return

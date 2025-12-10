"""
This module contains _functions to conduct data analysis prior to machine
learning model. Data transformation _functions are used if issues are present
and the user permits.
"""

# Built-Ins
import logging
from pathlib import Path
from typing import Optional

# Third Party
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import MissingDataError
import statsmodels.api as sm
from scipy.stats import pointbiserialr

# Local Imports
from caf.brain.ml._functions.process_data_functions.encode_and_scale import (
    preprocess_numerical_data,
)
from caf.brain.ml._functions._ml_inputs import PredictionModelInputs

LOG = logging.getLogger(__name__)


def pre_forecast_data_analysis(
    output_folder: Path,
    residuals: pd.Series,
    model_fit,
    model_initialised,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    train_unscaled: pd.DataFrame,
    test_unscaled: pd.DataFrame,
    numerical_pipeline: Optional[Pipeline],
    data_classification: Optional[PredictionModelInputs.DataClassificationInputs] = None,
    modelling: Optional[PredictionModelInputs.ModellingInputs] = None,
    target_column: Optional[str] = None,
    weight_column: Optional[str] = None,
    numerical_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
    is_time_series: Optional[bool] = None,
    allow_transformations: Optional[bool] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to conduct basic data analysis and fix where and if
    applicable (regression problems only).

    Parameters
    ----------
    data_classification: Data classification inputs from the PredictionModelInputs
                         class. These inputs help define and outline the
                         structure of the input data. See
                         caf/brain/ml/main_models/prediction_model/_ml_inputs.py
                         for available options.
    modelling: Modelling inputs from the PredictionModelInputs
               class. These inputs control the machine learning modelling
               pipeline and _functions. See
               caf/brain/ml/main_models/prediction_model/_ml_inputs.py
               for available options.
    residuals: Truth values form the train_test_split against the predictions.
    model_fit: Fitted model on train_test_split test data.
    model_initialised: Initialised SciKitLearn model.
    x_test: Dataframe of test data to be used as unseen test data.
    y_test: Dataframe of target data from train, test split.
    train_scaled: Processed input data split into train set.
    test_scaled: Processed input data split into train set.
    train_unscaled: Input data unprocessed split into train.
    test_unscaled: Input data unprocessed split into test.
    numerical_pipeline: Stored numerical transformation pipeline for
                        full model runs. Left as None if not a full
                        model run.
    output_folder: Path to output folder.
    target_column: Name of the target variable column
                   (used if data_classification not provided).
    weight_column: Name of the weight column
                   (used if data_classification not provided).
    numerical_features: List of numerical feature names
                        (used if data_classification not provided).
    categorical_features: List of categorical feature names
                          (used if data_classification not provided).
    is_time_series: Whether data is time series
                    (used if data_classification not provided).
    allow_transformations: Whether to apply transformations
                           (used if modelling not provided).

    Returns
    -------
    Train and test data post data transformations are returned
    if transformations permitted otherwise train and test scaled are returned.
    """

    # extract values from configs if provided
    _target_column = (
        data_classification.target_column if data_classification else target_column
    )
    _weight_column = (
        data_classification.weight_column if data_classification else weight_column
    )
    _numerical_features = (
        data_classification.numerical_features if data_classification else numerical_features
    )
    _categorical_features = (
        data_classification.categorical_features
        if data_classification
        else categorical_features
    )
    _is_time_series = (
        data_classification.is_time_series
        if data_classification
        else (is_time_series or False)
    )
    _allow_transformations = (
        modelling.full_transformations if modelling else (allow_transformations or False)
    )

    alpha = 0.05
    issues = {
        "linearity": False,
        "normality": False,
        "multicollinearity": False,
        "autocorrelation": False,
        "heteroscedasticity": False,
    }

    is_statsmodel = any(
        base.__module__.startswith("statsmodels")
        for base in model_initialised.__class__.__mro__
    )
    is_classification = isinstance(
        model_fit,
        (
            LogisticRegression,
            GradientBoostingClassifier,
            RandomForestClassifier,
            ExtraTreesClassifier,
            DecisionTreeClassifier,
            OneVsRestClassifier,
        ),
    ) or (
        is_statsmodel
        and ("Logit" in str(model_fit.__class__) or "MNLogit" in str(model_fit.__class__))
    )

    is_linear_model = isinstance(
        model_fit, (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression)
    ) or (
        is_statsmodel
        and any(
            name in str(model_fit.__class__) for name in ["OLS", "GLM", "Logit", "MNLogit"]
        )
    )

    if is_classification:
        raise ValueError(
            "Incorrect data evaluation selected for the modelling choice \n"
            "You are using classification meaning pre_forecast_data_analysis_classification \n"
            "should be used instead."
        )

    # multicollinearity
    if _numerical_features and len(_numerical_features) > 1:
        LOG.info("Checking for multicollinearity")
        x_num = train_scaled[numerical_features].copy()

        x_num = x_num.loc[:, x_num.std() > 1e-10]

        if x_num.shape[1] >= 2:
            try:
                vif_data = pd.DataFrame()
                vif_data["Feature"] = x_num.columns
                vif_data["VIF"] = [
                    variance_inflation_factor(x_num.values, i) for i in range(x_num.shape[1])
                ]
                high_vif = vif_data[vif_data["VIF"] > 10]
                if not high_vif.empty:
                    LOG.warning("High VIF variables: %s", high_vif)
                    issues["multicollinearity"] = True
                    vif_data.to_csv(output_folder / "vif_results.csv", index=False)
            except (ValueError, np.linalg.LinAlgError) as e:
                LOG.warning("VIF calculation failed: %s", e)
        else:
            LOG.info("Insufficient numerical features for VIF calculation (need >= 2)")
    elif _numerical_features:
        LOG.info("Only one numerical feature - skipping multicollinearity check")

    # Heteroscedasticity
    min_features_for_hetero_test = 3
    if x_test.shape[1] >= min_features_for_hetero_test:
        LOG.info("Checking for heteroscedasticity")

        try:
            x_with_const = add_constant(x_test)

            # Check if design matrix is full rank
            matrix_rank = np.linalg.matrix_rank(x_with_const)
            expected_rank = x_with_const.shape[1]

            if matrix_rank < expected_rank:
                LOG.warning(
                    "Design matrix is rank deficient (rank %d, expected %d). \n"
                    "This is likely due to perfect multicollinearity from encoded categorical variables. \n"
                    "Skipping heteroscedasticity tests.",
                    matrix_rank, expected_rank
                )
            else:
                # Breusch-Pagan
                try:
                    _, bp_test_p_value, _, _ = het_breuschpagan(residuals, x_with_const)
                    LOG.info("Breusch-Pagan test p-value: %s", bp_test_p_value)
                except (ValueError, MissingDataError, AssertionError) as e:
                    LOG.warning("Breusch-Pagan test failed: %s", e)
                    bp_test_p_value = 1.0

                # White Test
                try:
                    ols_model = sm.OLS(y_test, x_with_const).fit()
                    _, white_test_p_value, _, _ = het_white(ols_model.resid, ols_model.model.exog)
                    LOG.info("White's test p-value: %s", white_test_p_value)
                except (ValueError, MissingDataError, AssertionError, np.linalg.LinAlgError) as e:
                    LOG.warning("White's test failed: %s", e)
                    white_test_p_value = 1.0

                if bp_test_p_value < alpha or white_test_p_value < alpha:
                    LOG.warning("Heteroscedasticity detected.")
                    issues["heteroscedasticity"] = True

        except Exception as e:
            LOG.warning("Heteroscedasticity tests failed: %s", e)
    else:
        LOG.info(
            "Skipping heteroscedasticity tests: insufficient features."
        )

    if is_linear_model and residuals is not None:
        LOG.info("Running tests for linear model assumptions")

        # Linearity
        LOG.info("Checking linearity")
        for col in x_test.columns:
            try:
                correlation = np.corrcoef(x_test[col], residuals)[0, 1]
                if abs(correlation) > 0.1:
                    LOG.warning("Warning: %s may not be linearly related to the target.", col)
                    issues["linearity"] = True
            except (ValueError, MissingDataError) as e:
                LOG.warning("Linearity test failed for column %s: %s", col, e)

        # Normality
        try:
            if len(residuals) > 5000:
                LOG.info(
                    "Large sample size (%d) - Shapiro-Wilk may be overly sensitive. \n"
                    "Consider results cautiously.", len(residuals)
                )

            _, shapiro_p_value = shapiro(residuals)
            LOG.info("Shapiro-Wilk test p-value: %s", shapiro_p_value)
            if shapiro_p_value < alpha:
                LOG.warning("Warning: Shapiro-Wilk test suggests non-normality of residuals.")
                issues["normality"] = True
        except (ValueError, MissingDataError) as e:
            LOG.warning("Shapiro-Wilk test failed: %s", e)

    if _is_time_series and residuals is not None:
        # Autocorrelation
        try:
            dw_statistic = durbin_watson(residuals)
            LOG.info("Durbin-Watson statistic: %s", dw_statistic)
            if dw_statistic < 1.5 or dw_statistic > 2.5:
                LOG.warning("Warning: Potential autocorrelation in residuals.")
                issues["autocorrelation"] = True
        except (ValueError, MissingDataError) as e:
            LOG.warning("Durbin-Watson test failed: %s", e)

    train_out, test_out = train_scaled, test_scaled
    issues_df = pd.DataFrame(list(issues.items()), columns=["test", "result"])

    if any(issues.values()):
        LOG.warning("Data issue present: %s", issues_df)
        issues_df.to_csv(output_folder / "data_issues_present.csv", index=False)

        if (
            _allow_transformations
            and _numerical_features
            and train_unscaled is not None
            and test_unscaled is not None
        ):

            LOG.info("Transformations applied to numerical data to fix the issues")
            train_out = transform_data(
                df=train_unscaled,
                is_test_data=False,
                numerical_pipeline=numerical_pipeline,
                output_folder=output_folder,
                target_column=_target_column,
                weight_column=_weight_column,
                numerical_features=_numerical_features,
                categorical_features=_categorical_features,
            )
            test_out = transform_data(
                df=test_unscaled,
                is_test_data=True,
                numerical_pipeline=numerical_pipeline,
                output_folder=output_folder,
                target_column=_target_column,
                weight_column=_weight_column,
                numerical_features=_numerical_features,
                categorical_features=_categorical_features,
            )
        else:
            LOG.warning(
                "Data issue present but no numerical features are present or full transformations have \
                 not been permitted so transformations can't occur"
            )
    else:
        LOG.info("No data issues present.")

    return train_out, test_out


def transform_data(
    df: pd.DataFrame,
    is_test_data: bool,
    numerical_pipeline,
    output_folder: Path,
    target_column: Optional[str],
    weight_column: Optional[str],
    numerical_features: list[str],
    categorical_features: Optional[list[str]],
) -> pd.DataFrame:
    """
    Function to apply data transformations.

    Parameters
    ----------
    df: Processed input data split into train or test set.
    is_test_data: True if test data being passed.
    numerical_pipeline: Stored numerical transformation pipeline for
                        full model runs. Left as None if not a full
                        model run.
    output_folder: Path to output folder.
    target_column: Name of the target variable column.
    weight_column: Name of the weight column if present.
    numerical_features: List of numerical feature column names.
    categorical_features: List of categorical feature column names.

    Returns
    -------
    transformed_df: Transformed data.
    """

    transformed_data = []

    original_index = df.index

    if target_column and target_column in df.columns:
        target = df[target_column].copy()
    else:
        target = None
    if weight_column and weight_column in df.columns:
        weight = df[weight_column].copy()
    else:
        weight = None

    numerical_data = df[numerical_features].copy()

    # log
    numerical_transformed = np.log1p(numerical_data)

    # force fixing any issues post log transformations
    numerical_transformed = numerical_transformed.replace([np.inf, -np.inf], np.nan)
    numerical_transformed = numerical_transformed.fillna(numerical_transformed.mean())

    # scale
    numerical_scaled, _ = preprocess_numerical_data(
        df=numerical_transformed,
        numerical_features=numerical_features,
        is_test_data=is_test_data,
        numerical_pipeline_train=numerical_pipeline,
        output_folder=output_folder,
    )

    transformed_data.append(numerical_scaled)

    if categorical_features is not None:
        categorical_data = df[categorical_features].copy()
        transformed_data.append(categorical_data)

    if weight is not None:
        transformed_data.append(weight)

    transformed_df = pd.concat(transformed_data, axis=1)

    if target is not None:
        transformed_df[target_column] = target
        transformed_df[target_column] = transformed_df[target_column].astype(int)

    transformed_df.index = original_index

    return transformed_df


def pre_forecast_data_analysis_classification(
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    train_unscaled: pd.DataFrame,
    target: str,
    numerical_features: list[str],
    output_path: Path,
    classification_prediction: tuple[int, ...] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to conduct basic data analysis prior to modelling for
    classification problems only.

    Parameters
    ----------
    train_scaled: Processed input data split into train set.
    test_scaled: Processed input data split into train set.
    train_unscaled: Input data unprocessed split into train.
    target: Name of the target variable column
    numerical_features: List of numerical feature names
    output_path: Path to output folder.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    Returns
    -------
    Train and test scaled are returned (same as the input files as no
    transformations are applied).
    """

    if len(classification_prediction) > 2:
        classification_type = "multiclass"
    else:
        classification_type = "binary"

    issues = {
        "multicollinearity": False,
        "class_imbalance": False,
        "weak_features": False,
    }

    # multicollinearity
    if numerical_features and len(numerical_features) > 1:
        LOG.info("Checking for multicollinearity")
        x_num = train_scaled[numerical_features].copy()

        x_num = x_num.loc[:, x_num.std() > 1e-10]

        if x_num.shape[1] >= 2:
            try:
                vif_data = pd.DataFrame()
                vif_data["Feature"] = x_num.columns
                vif_data["VIF"] = [
                    variance_inflation_factor(x_num.values, i) for i in range(x_num.shape[1])
                ]
                high_vif = vif_data[vif_data["VIF"] > 10]
                if not high_vif.empty:
                    LOG.warning("High VIF variables: %s", high_vif)
                    issues["multicollinearity"] = True
                    vif_data.to_csv(output_path / "vif_results.csv", index=False)
            except (ValueError, np.linalg.LinAlgError) as e:
                LOG.warning("VIF calculation failed: %s", e)
        else:
            LOG.info("Insufficient numerical features for VIF calculation (need >= 2)")
    elif numerical_features:
        LOG.info("Only one numerical feature - skipping multicollinearity check")


    # class imbalance
    class_counts = train_unscaled[target].value_counts()
    class_ratios = (class_counts / len(train_unscaled)).to_dict()
    min_ratio = min(class_ratios.values())
    max_ratio = max(class_ratios.values())

    imbalance_ratio = max_ratio / min_ratio

    if min_ratio < 0.1 or imbalance_ratio > 3:
        LOG.warning("Class imbalance detected. Ratios: %s", class_ratios)
        LOG.warning("Consider: SMOTE, class weights, or stratified sampling")
        issues["class_imbalance"] = True

    class_balance_df = pd.DataFrame(
        {
            "class": class_counts.index,
            "count": class_counts.values,
            "ratio": [class_ratios[c] for c in class_counts.index],
        }
    )
    class_balance_df.to_csv(output_path / "class_balance.csv", index=False)

    # feature target corr
    if classification_type == "binary" and numerical_features:

        weak_features = []
        correlations = []

        for col in numerical_features:
            try:
                corr, pval = pointbiserialr(train_unscaled[target], train_unscaled[col])
                correlations.append({"feature": col, "correlation": corr, "p_value": pval})
                if abs(corr) < 0.05:
                    if pval > 0.05:
                        weak_features.append(col)
            except Exception as e:
                LOG.warning("Feature-target correlation failed for %s: %s", col, e)
                continue

        if weak_features:
            LOG.warning("Weak feature target correlation in: %s", weak_features)
            issues["weak_features"] = True

        corr_df = pd.DataFrame(correlations)
        corr_df.to_csv(output_path / "feature_target_correlations.csv", index=False)

    issues_df = pd.DataFrame(list(issues.items()), columns=["test", "issue_detected"])
    issues_df.to_csv(output_path / "classification_diagnostics.csv", index=False)

    if any(issues.values()):
        LOG.warning("Issues detected in classification data. Check output files for details.")
    else:
        LOG.info("No major issues detected in classification data.")

    LOG.info("Classification data evaluation complete. Results saved to %s", output_path)

    return train_scaled, test_scaled

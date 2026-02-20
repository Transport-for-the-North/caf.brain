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
import statsmodels.api as sm
from scipy.stats import chi2_contingency, f_oneway, pointbiserialr, shapiro
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

# Local Imports
from caf.brain.ml._functions._ml_inputs import DataClassificationInputs, ModellingInputs
from caf.brain.ml._functions.process_data_functions.encode_and_scale import (
    preprocess_numerical_data,
)

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
    data_classification: Optional[DataClassificationInputs] = None,
    modelling: Optional[ModellingInputs] = None,
    target_column: Optional[str] = None,
    weight_column: Optional[str] = None,
    numerical_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
    is_time_series: Optional[bool] = None,
    allow_transformations: Optional[bool] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conduct data analysis for regression or classification problems.

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
        needs_transformations = _classification_only(
            is_classification=is_classification,
            numerical_features=_numerical_features,
            categorical_features=_categorical_features,
            df=train_unscaled,
            target=_target_column,
            output_path=output_folder,
        )

    else:
        needs_transformations = _regression_only(
            numerical_features=_numerical_features,
            categorical_features=_categorical_features,
            df=train_unscaled,
            target=_target_column,
            output_path=output_folder,
            x_test=x_test,
            y_test=y_test,
            residuals=residuals,
            is_linear_model=is_linear_model,
            is_time_series=is_time_series,
        )

    if needs_transformations:
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
                "Data issue present but no numerical features are present or full transformations have \n"
                "not been permitted so transformations can't occur"
            )
            train_out = train_scaled
            test_out = test_scaled
    else:
        LOG.info("No data issue present")
        train_out = train_scaled
        test_out = test_scaled

    return train_out, test_out


def _classification_only(
    is_classification: bool,
    numerical_features: list[str] | None,
    categorical_features: list[str] | None,
    df: pd.DataFrame,
    target: str | None,
    output_path: Path,
) -> bool:
    """
    Classification only problems data analysis.

    Parameters
    ----------
    is_classification:
        True if classification prediction.
    numerical_features:
        List of numerical feature names.
    categorical_features:
        List of categorical feature names.
    df:
        Dataframe containing the data (not encoded or scaled).
    target:
        Name of the target variable column.
    output_path:
        Path to output folder.

    Returns
    -------
    True if issues detected.
    """

    if target is None:
        raise ValueError(
            "Target column must be provided. Ensure target is \n"
            "populated if using data_classification.target in \n"
            "the config file. If calling the pre_forecast_data_analysis\n"
            "function directly, ensure target column is provided."
        )

    numerical_to_fix = []
    categorical_warnings = []
    all_issues = []
    full_path = output_path / "data_analysis_output"
    full_path.mkdir(parents=True, exist_ok=True)

    # NUMERICAL FEATURE CHECKS
    if is_classification and numerical_features:
        if len(set(df[target])) == 2:
            has_issue, weak_feats = _check_numerical_feat_class_target(
                numerical_features=numerical_features,
                df=df,
                target=target,
                output_path=full_path,
            )
            if has_issue:
                numerical_to_fix.extend(weak_feats)
                all_issues.append(
                    {
                        "issue_type": "Weak Numerical-Target Correlation",
                        "severity": "HIGH - Will be transformed",
                        "affected_features": ", ".join(weak_feats),
                        "recommendation": "Log transformation will be applied (if permitted)",
                    }
                )

    if numerical_features:
        has_issue, high_vif_feats = _multicolinearity_check(
            df=df, numerical_features=numerical_features, output_path=full_path
        )
        if has_issue:
            numerical_to_fix.extend(high_vif_feats)
            all_issues.append(
                {
                    "issue_type": "Multicollinearity (VIF > 10)",
                    "severity": "HIGH - Will be transformed",
                    "affected_features": ", ".join(high_vif_feats),
                    "recommendation": "Log transformation will be applied (if permitted)",
                }
            )

    # CATEGORICAL FEATURE CHECKS
    has_imbalance = _class_imbalance_classification_target(
        df=df, target_column=target, output_path=full_path
    )
    if has_imbalance:
        all_issues.append(
            {
                "issue_type": "Class Imbalance",
                "severity": "WARNING - Manual action recommended",
                "affected_features": target,
                "recommendation": "Consider SMOTE, class weights, or stratified sampling",
            }
        )

    if categorical_features:
        has_issue, weak_feats = _check_categorical_classification(
            df=df,
            target=target,
            categorical_features=categorical_features,
            output_path=full_path,
        )
        if has_issue:
            categorical_warnings.extend(weak_feats)
            all_issues.append(
                {
                    "issue_type": "Weak Categorical-Target Relationship (Chi-square p > 0.05)",
                    "severity": "WARNING - Manual action recommended",
                    "affected_features": ", ".join(weak_feats),
                    "recommendation": "Consider feature engineering or removal",
                }
            )

    if categorical_features:
        has_issue, rare_feats = _check_rare_categories(
            df=df, categorical_features=categorical_features, output_path=full_path
        )
        if has_issue:
            categorical_warnings.extend(rare_feats)
            all_issues.append(
                {
                    "issue_type": "Rare Categories (< 5% frequency)",
                    "severity": "WARNING - Manual action recommended",
                    "affected_features": ", ".join(rare_feats),
                    "recommendation": "Consider grouping rare categories or removal",
                }
            )

    if categorical_features:
        has_issue, high_card_feats = _check_cardinality(
            df=df, categorical_features=categorical_features, output_path=full_path
        )
        if has_issue:
            categorical_warnings.extend(high_card_feats)
            all_issues.append(
                {
                    "issue_type": "High Cardinality (> 50 unique values)",
                    "severity": "WARNING - Manual action recommended",
                    "affected_features": ", ".join(high_card_feats),
                    "recommendation": "Consider dimensionality reduction",
                }
            )

    if all_issues:
        pd.DataFrame(all_issues).to_csv(full_path / "data_issues_summary.csv", index=False)
        LOG.warning(
            "Data issues detected. See data_issues_summary.csv in %s for full summary.",
            full_path,
        )
    else:
        LOG.info("No data issues detected.")

    if len(categorical_warnings) > 0:
        categorical_detail = pd.DataFrame(
            {
                "feature": list(set(categorical_warnings)),
                "issue_count": [
                    categorical_warnings.count(feat) for feat in set(categorical_warnings)
                ],
            }
        )
        categorical_detail.to_csv(full_path / "categorical_warnings_detail.csv", index=False)

    return len(all_issues) > 0


def _regression_only(
    df: pd.DataFrame,
    target: str | None,
    numerical_features: list[str] | None,
    categorical_features: list[str] | None,
    output_path: Path,
    x_test: pd.DataFrame = None,
    y_test: pd.DataFrame = None,
    residuals: pd.Series = None,
    is_linear_model: bool = False,
    is_time_series: bool | None = False,
) -> bool:
    """
    Regression only problems data analysis.

    Parameters
    ----------
    target:
        Name of the target variable column.
    output_path:
        Path to output folder.
    numerical_features:
        List of numerical feature names.
    categorical_features:
        List of categorical feature names.
    df:
        Dataframe containing the data (not encoded or scaled).
    x_test:
        Dataframe of test data to be used as unseen test data.
    y_test:
        Dataframe of target data from train, test split.
    residuals:
        Truth values form the train_test_split against the predictions.
    is_linear_model:
        Whether the fitted model is a linear regression model.
    is_time_series:
        If data is temporal in nature.

    Returns
    -------
    True if issues detected.
    """

    if target is None:
        raise ValueError(
            "Target column must be provided. Ensure target is \n"
            "populated if using data_classification.target in \n"
            "the config file. If calling the pre_forecast_data_analysis\n"
            "function directly, ensure target column is provided."
        )

    numerical_to_fix = []
    categorical_warnings = []
    all_issues = []
    full_path = output_path / "data_analysis_output"
    full_path.mkdir(parents=True, exist_ok=True)

    # NUMERICAL FEATURE CHECKS
    if numerical_features:
        has_issue, high_vif_feats = _multicolinearity_check(
            df=df, numerical_features=numerical_features, output_path=full_path
        )
        if has_issue:
            numerical_to_fix.extend(high_vif_feats)
            all_issues.append(
                {
                    "issue_type": "Multicollinearity (VIF > 10)",
                    "severity": "HIGH - Will be transformed",
                    "affected_features": ", ".join(high_vif_feats),
                    "recommendation": "Log transformation will be applied (if permitted)",
                }
            )

    # LINEAR MODEL CHECKS
    if is_linear_model:
        heteroscedasticity_flag = _heteroscedasticity_check(
            x_test=x_test,
            y_test=y_test,
            numerical_features=numerical_features,
            residuals=residuals,
            is_linear_model=is_linear_model,
        )

        linear_model_issues = _linear_model_tests(
            x_test=x_test,
            numerical_features=numerical_features,
            is_time_series=is_time_series,
            residuals=residuals,
            is_linear_model=is_linear_model,
        )

        if heteroscedasticity_flag:
            categorical_warnings.append("Heteroscedasticity")
            all_issues.append(
                {
                    "issue_type": "Heteroscedasticity",
                    "severity": "WARNING - Linear model assumption violated",
                    "affected_features": "All numerical features",
                    "recommendation": "Consider transformation. Applied if permitted",
                }
            )

        if linear_model_issues:
            categorical_warnings.append("Linear model assumptions violated")
            all_issues.append(
                {
                    "issue_type": "Linear Model Assumptions (linearity/normality/autocorrelation)",
                    "severity": "WARNING - Linear model assumption violated",
                    "affected_features": "Model residuals",
                    "recommendation": "Review diagnostic plots; consider non-linear models",
                }
            )

    # CATEGORICAL FEATURE CHECKS
    if categorical_features:
        has_issue, weak_feats = _check_categorical_regression(
            df=df,
            target=target,
            categorical_features=categorical_features,
            output_path=full_path,
        )
        if has_issue:
            categorical_warnings.extend(weak_feats)
            all_issues.append(
                {
                    "issue_type": "Weak Categorical-Target Relationship (ANOVA p > 0.05)",
                    "severity": "WARNING - Manual action recommended",
                    "affected_features": ", ".join(weak_feats),
                    "recommendation": "Consider feature engineering or removal",
                }
            )

        has_issue, rare_feats = _check_rare_categories(
            df=df, categorical_features=categorical_features, output_path=full_path
        )
        if has_issue:
            categorical_warnings.extend(rare_feats)
            all_issues.append(
                {
                    "issue_type": "Rare Categories (< 5% frequency)",
                    "severity": "WARNING - Manual action recommended",
                    "affected_features": ", ".join(rare_feats),
                    "recommendation": "Consider grouping rare categories or removal",
                }
            )

        has_issue, high_card_feats = _check_cardinality(
            df=df, categorical_features=categorical_features, output_path=full_path
        )
        if has_issue:
            categorical_warnings.extend(high_card_feats)
            all_issues.append(
                {
                    "issue_type": "High Cardinality (> 50 unique values)",
                    "severity": "WARNING - Manual action recommended",
                    "affected_features": ", ".join(high_card_feats),
                    "recommendation": "Consider dimensionality reduction",
                }
            )

    if all_issues:
        pd.DataFrame(all_issues).to_csv(full_path / "data_issues_summary.csv", index=False)
        LOG.warning(
            "Data issues detected. See data_issues_summary.csv in %s for full summary.",
            full_path,
        )
    else:
        LOG.info("No data issues detected.")

    if len(categorical_warnings) > 0:
        categorical_detail = pd.DataFrame({"warning": list(set(categorical_warnings))})
        categorical_detail.to_csv(full_path / "categorical_warnings_detail.csv", index=False)

    return len(all_issues) > 0


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


def _linear_model_tests(
    x_test: pd.DataFrame,
    numerical_features: list[str] | None,
    is_time_series: bool | None = False,
    residuals: pd.Series = None,
    is_linear_model: bool | None = False,
) -> bool:
    """
    Perform diagnostic checks for linear regression model assumptions.

    This function evaluates three key OLS assumptions:
        - Linearity: whether numerical predictors have a linear relationship
          with the target (checked via correlation with residuals).
        - Normality: whether residuals follow a normal distribution
          (Shapiro–Wilk test).
        - Autocorrelation: whether residuals are independent over time
          (Durbin–Watson test; only applied for time-series data).

    Only applicable to linear regression models with numerical predictors and
    continuous residuals.

    Parameters
    ----------
    x_test:
        Test-set feature matrix.
    numerical_features:
        List of numerical features.
    is_time_series:
        Whether the data has temporal ordering.
    residuals:
        Residuals from the fitted linear model.
    is_linear_model:
        Whether the fitted model is a linear regression model.

    Returns
    -------
    has_violations:
        True if any of the linear model assumptions (linearity, normality,
        autocorrelation) appear to be violated, otherwise False.
    """

    if residuals is None:
        LOG.warning("Residual data is not available, skipping linear model tests.")
        return False

    if not is_linear_model:
        LOG.warning("Model is not linear, skipping linear model tests.")
        return False

    if not numerical_features:
        LOG.info("Numerical features are required, linear model tests, skipping.")
        return False

    if is_linear_model and residuals is not None:
        LOG.info("Running tests for linear model assumptions")
        linearity = False
        normality = False
        autocorrelation = False

        # Linearity
        LOG.info("Checking linearity")
        for col in x_test.columns:
            try:
                correlation = np.corrcoef(x_test[col], residuals)[0, 1]
                if abs(correlation) > 0.1:
                    linearity = True
                    LOG.warning("Warning: %s may not be linearly related to the target.", col)
            except (ValueError, MissingDataError) as e:
                LOG.warning("Linearity test failed for column %s: %s", col, e)

        # Normality
        try:
            if len(residuals) > 5000:
                LOG.info(
                    "Large sample size (%d) - Shapiro-Wilk may be overly sensitive. \n"
                    "Consider results cautiously.",
                    len(residuals),
                )

            _, shapiro_p_value = shapiro(residuals)
            LOG.info("Shapiro-Wilk test p-value: %s", shapiro_p_value)
            if shapiro_p_value < 0.05:
                LOG.warning("Warning: Shapiro-Wilk test suggests non-normality of residuals.")
                normality = True
        except (ValueError, MissingDataError) as e:
            LOG.warning("Shapiro-Wilk test failed: %s", e)

        if is_time_series:
            # Autocorrelation
            try:
                dw_statistic = durbin_watson(residuals)
                LOG.info("Durbin-Watson statistic: %s", dw_statistic)
                if dw_statistic < 1.5 or dw_statistic > 2.5:
                    LOG.warning("Warning: Potential autocorrelation in residuals.")
                    autocorrelation = True
            except (ValueError, MissingDataError) as e:
                LOG.warning("Durbin-Watson test failed: %s", e)

        if autocorrelation or normality or linearity:
            LOG.warning("Linear model specific issues detected.")
            return True

    return False


def _heteroscedasticity_check(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    numerical_features: list[str] | None,
    residuals: pd.Series = None,
    is_linear_model: bool = False,
) -> bool:
    """
    Test for heteroscedasticity using the Breusch–Pagan and White tests.

    Heteroscedasticity refers to non-constant variance of residuals and is a
    violation of the assumptions of ordinary least squares (OLS) regression.
    This diagnostic is only applicable to linear models with numerical predictors.

    Parameters
    ----------
    x_test:
        Test-set feature matrix.
    y_test:
        Test-set target values.
    numerical_features:
        List of numerical features.
    residuals:
        Residuals from the fitted linear model. Required for the tests.
    is_linear_model:
        Whether the fitted model is a linear regression model.

    Returns
    -------
    has_heteroscedasticity:
        True if either the Breusch–Pagan or White test detects heteroscedasticity
        (p < 0.05), otherwise False.
    """

    if residuals is None:
        LOG.warning("Residual data is not available for heteroscedasticity, skipping test.")
        return False

    if not is_linear_model:
        LOG.warning("Model is not linear, skipping heteroscedasticity test.")
        return False

    min_features_for_hetero_test = 3
    if x_test.shape[1] <= min_features_for_hetero_test:
        LOG.info("Skipping heteroscedasticity tests: insufficient features.")
        return False

    if not numerical_features:
        LOG.info(
            "Numerical features are required for heteroscedasticity tests, skipping heteroscedasticity test."
        )
        return False

    LOG.info("Checking for heteroscedasticity")
    x_test_copy = x_test[numerical_features].copy()

    try:
        x_with_const = add_constant(x_test_copy)

        # Check if design matrix is full rank
        matrix_rank = np.linalg.matrix_rank(x_with_const)
        expected_rank = x_with_const.shape[1]

        if matrix_rank < expected_rank:
            LOG.warning(
                "Design matrix is rank deficient (rank %d, expected %d). \n"
                "This is likely due to perfect multicollinearity from encoded categorical variables. \n"
                "Skipping heteroscedasticity tests.",
                matrix_rank,
                expected_rank,
            )
        else:
            try:
                _, bp_test_p_value, _, _ = het_breuschpagan(residuals, x_with_const)
                LOG.info("Breusch-Pagan test p-value: %s", bp_test_p_value)
            except (ValueError, MissingDataError, AssertionError) as e:
                LOG.warning("Breusch-Pagan test failed: %s", e)
                bp_test_p_value = 1.0

            try:
                ols_model = sm.OLS(y_test, x_with_const).fit()
                _, white_test_p_value, _, _ = het_white(ols_model.resid, ols_model.model.exog)
                LOG.info("White's test p-value: %s", white_test_p_value)
            except (
                ValueError,
                MissingDataError,
                AssertionError,
                np.linalg.LinAlgError,
            ) as e:
                LOG.warning("White's test failed: %s", e)
                white_test_p_value = 1.0

            if bp_test_p_value < 0.05 or white_test_p_value < 0.05:
                LOG.warning("Heteroscedasticity detected.")
                return True

    except Exception as e:
        LOG.warning("Heteroscedasticity tests failed: %s", e)
    return False


def _multicolinearity_check(
    df: pd.DataFrame, numerical_features: list[str], output_path: Path
) -> tuple[bool, list[str]]:
    """
    Check numerical features for multicollinearity with Variance Inflation Factor.

    A feature is considered problematic if its VIF exceeds 10, indicating strong
    multicollinearity with other numerical predictors.

    Any problem type, just for numerical x variables.

    Parameters
    ----------
    df:
        Input dataframe containing numerical features.
    numerical_features:
        List of numerical feature names to evaluate.
    output_path:
        Directory where the VIF results CSV will be saved.

    Returns
    -------
    has_multicollinearity:
        True if any features exhibit high VIF (>10), otherwise False.
    high_vif_features:
        List of numerical features with VIF greater than 10.
    """

    high_vif_features: list[str] = []

    if not numerical_features or len(numerical_features) < 2:
        LOG.info("Insufficient numerical features for VIF calculation (need >= 2)")
        return False, high_vif_features

    LOG.info("Checking for multicollinearity")

    x_num = df[numerical_features].copy()
    x_num = x_num.loc[:, x_num.std() > 1e-10]

    if x_num.shape[1] < 2:
        LOG.info("After removing near-constant columns, insufficient features for VIF")
        return False, high_vif_features

    try:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = x_num.columns
        vif_data["VIF"] = [
            variance_inflation_factor(x_num.values, i) for i in range(x_num.shape[1])
        ]

        high_vif = vif_data[vif_data["VIF"] > 10]

        if not high_vif.empty:
            high_vif_features = high_vif["Feature"].tolist()
            LOG.warning("High VIF variables detected: %s", high_vif_features)

        vif_data.to_csv(output_path / "vif_results.csv", index=False)

        return len(high_vif_features) > 0, high_vif_features

    except (ValueError, np.linalg.LinAlgError) as e:
        LOG.warning("VIF calculation failed: %s", e)
        return False, high_vif_features


def _class_imbalance_classification_target(
    df: pd.DataFrame, target_column: str, output_path: Path
) -> bool:
    """
    Check for class imbalance in a classification target.

    Only for classification problems.

    Parameters
    ----------
    df:
        Input dataframe containing the target column.
    target_column:
        Name of the target variable.
    output_path:
        Path to output folder.

    Returns
    -------
    has_imbalance:
        True if class imbalance is detected, otherwise False.
    """

    class_counts = df[target_column].value_counts()
    class_ratios = (class_counts / len(df)).to_dict()
    min_ratio = min(class_ratios.values())
    max_ratio = max(class_ratios.values())

    imbalance_ratio = max_ratio / min_ratio

    has_imbalance = (min_ratio < 0.1) or (imbalance_ratio > 3)
    if has_imbalance:
        LOG.warning("Class imbalance detected. Ratios: %s", class_ratios)
        LOG.warning("Consider: SMOTE, class weights, or stratified sampling")

    class_balance_df = pd.DataFrame(
        {
            "class": class_counts.index,
            "count": class_counts.values,
            "ratio": [class_ratios[c] for c in class_counts.index],
        }
    )
    class_balance_df.to_csv(output_path / "class_balance.csv", index=False)

    return has_imbalance


def _check_cardinality(
    df: pd.DataFrame, categorical_features: list[str], output_path: Path, threshold: int = 50
) -> tuple[bool, list[str]]:
    """
    Check for high cardinality in categorical variables.

    This is where you have too many categories for a categorical variable (e.g.
    1 feature with 2000 categories would cause 2000 dummy columns.).

    Categorical feature check (both problem types).

    Parameters
    ----------
    df:
        Input dataframe.
    categorical_features:
        List of categorical variables to check.
    output_path:
        Path to output file.
    threshold:
        Threshold to check for high cardinality.

    Returns
    -------
    Tuple (bool and the high_card_features)
    """

    high_card_features = []
    cardinality_report = []

    for col in categorical_features:
        n_unique = df[col].nunique()
        cardinality_report.append(
            {
                "feature": col,
                "unique_values": n_unique,
                "cardinality_ratio": n_unique / len(df),
            }
        )

        if n_unique > threshold:
            high_card_features.append(col)
            LOG.warning("High cardinality in %s: %s unique values", col, n_unique)

    pd.DataFrame(cardinality_report).to_csv(
        output_path / "cardinality_report.csv", index=False
    )

    return len(high_card_features) > 0, high_card_features


def _check_rare_categories(
    df: pd.DataFrame,
    categorical_features: list[str],
    output_path: Path,
    threshold: float = 0.05,
) -> tuple[bool, list[str]]:
    """
    Detect rare categories in categorical features.

    A rare category is defined as one whose frequency is below the given
    threshold (default = 5%).

    Categorical feature check (both problem types).

    Parameters
    ----------
    df:
        Input dataframe.
    categorical_features:
        List of categorical variables to check.
    output_path:
        Path to output file.
    threshold:
        Threshold to check for rarity.

    Returns
    -------
    has_rare_categories:
        True if any rare categories were found, otherwise False.
    rare_features:
        List of feature names that contain at least one rare category.
    """
    rare_cats_found = False
    rare_report = []

    for col in categorical_features:
        value_counts = df[col].value_counts(normalize=True)
        rare_values = value_counts[value_counts < threshold]

        if len(rare_values) > 0:
            rare_cats_found = True
            LOG.warning(
                "Rare categories in %s: %s categories <%s percent",
                col,
                len(rare_values),
                (threshold * 100),
            )

            for cat, freq in rare_values.items():
                rare_report.append(
                    {
                        "feature": col,
                        "category": cat,
                        "frequency": freq,
                        "count": (df[col] == cat).sum(),
                    }
                )

    if rare_report:
        pd.DataFrame(rare_report).to_csv(output_path / "rare_categories.csv", index=False)
    rare_features = list({row["feature"] for row in rare_report})
    return rare_cats_found, rare_features


def _check_categorical_classification(
    df: pd.DataFrame, target: str, categorical_features: list[str], output_path: Path
) -> tuple[bool, list[str]]:
    """
    Check the categorical feature versus categorical target relationship.

    Only for classification problem.

    Parameters
    ----------
    df:
        Input dataframe.
    target:
        Target column (y).
    categorical_features:
        List of categorical variables to check.
    output_path:
        Path to output file.

    Returns
    -------
    has_weak_features:
        True if any categorical features show a weak relationship with the
        target.
    weak_features:
        List of categorical feature names that failed the test.
    """
    weak_features = []
    chi_square_results = []

    for col in categorical_features:
        contingency_table = pd.crosstab(df[col], df[target])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        chi_square_results.append(
            {
                "feature": col,
                "chi2": chi2,
                "p_value": p_value,
            }
        )

        if p_value > 0.05:
            weak_features.append(col)
            LOG.warning("Weak relationship: %s vs %s (p=%.4f)", col, target, p_value)

    pd.DataFrame(chi_square_results).to_csv(
        output_path / "categorical_chi_square.csv", index=False
    )

    return len(weak_features) > 0, weak_features


def _check_categorical_regression(
    df: pd.DataFrame, target: str, categorical_features: list[str], output_path: Path
) -> tuple[bool, list[str]]:
    """
    One way anova test.

    Only for regression problems and categorical features.

    Parameters
    ----------
    df:
        Input dataframe.
    target:
        Target column (y).
    categorical_features:
        List of categorical variables to check.
    output_path:
        Path to output file.

    Returns
    -------
    has_weak_features:
        True if any categorical features show a weak relationship with the
        target.
    weak_features:
        List of categorical feature names that failed the test.
    """
    weak_features = []
    anova_results = []

    for col in categorical_features:
        groups = [df[df[col] == cat][target].values for cat in df[col].unique()]

        f_stat, p_value = f_oneway(*groups)

        anova_results.append(
            {
                "feature": col,
                "f_statistic": f_stat,
                "p_value": p_value,
            }
        )

        if p_value > 0.05:
            weak_features.append(col)
            LOG.warning("Weak relationship: %s vs %s (p=%.4f)", col, target, p_value)

    pd.DataFrame(anova_results).to_csv(output_path / "categorical_anova.csv", index=False)

    return len(weak_features) > 0, weak_features


def _check_numerical_feat_class_target(
    numerical_features: list[str], df: pd.DataFrame, target: str, output_path: Path
) -> tuple[bool, list[str]]:
    """
    Evaluate relationship between numerical features and a binary classification
    target.

    Numerical features and binary target only.

    Parameters
    ----------
    numerical_features:
        List of numerical features.
    df:
        Input dataframe.
    target:
        Name of the target column.
    output_path:
        Path to output file.

    Returns
    -------
    has_weak_features:
        True if any numerical features show weak correlation with the target.
    weak_features:
        List of numerical feature names identified as weak predictors.
    """
    weak_features = []
    correlations = []

    for col in numerical_features:
        try:
            corr, pval = pointbiserialr(df[target], df[col])
            correlations.append({"feature": col, "correlation": corr, "p_value": pval})
            if abs(corr) < 0.05:
                if pval > 0.05:
                    weak_features.append(col)
        except Exception as e:
            LOG.warning("Feature-target correlation failed for %s: %s", col, e)
            continue

    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(output_path / "feature_target_correlations.csv", index=False)

    return len(weak_features) > 0, weak_features

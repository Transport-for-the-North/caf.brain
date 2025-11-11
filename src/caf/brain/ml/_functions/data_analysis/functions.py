"""
This module contains functions to conduct data analysis prior to machine
learning model. Data transformation functions are used if issues are present
and the user permits.
"""

# Built-Ins
import logging
from pathlib import Path

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
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import MissingDataError

# Local Imports
# from sklearn.decomposition import PCA
from caf.brain.ml._functions.process_data_functions.encode_and_scale import (
    preprocess_numerical_data,
)
from caf.brain.ml._inputs_and_baseclasses.ml_inputs import PredictionModelInputs

LOG = logging.getLogger(__name__)


def pre_forecast_data_analysis(
    data_classification: PredictionModelInputs.DataClassificationInputs,
    modelling: PredictionModelInputs.ModellingInputs,
    output_folder: Path,
    residuals: pd.Series,
    model_fit,
    model_initialised,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    train_unscaled: pd.DataFrame,
    test_unscaled: pd.DataFrame,
    numerical_pipeline,
):
    """
    Function to conduct basic data analysis and fix where and if
    applicable.

    Parameters
    ----------
    data_classification: Data classification inputs from the PredictionModelInputs
                         class. These inputs help define and outline the
                         structure of the input data. See
                         caf/brain/ml/main_models/prediction_model/ml_inputs.py
                         for available options.
    modelling: Modelling inputs from the PredictionModelInputs
               class. These inputs control the machine learning modelling
               pipeline and functions. See
               caf/brain/ml/main_models/prediction_model/ml_inputs.py
               for available options.
    residuals: Truth values form the train_test_split against the predictions.
    model_fit: Fitted model on train_test_split test data.
    model_initialised: Initialised SciKitLearn model.
    x_train: Series of train data to be used as train.
    x_test: Series of test data to be used as unseen test data.
    train_scaled: Processed input data split into train set.
    test_scaled: Processed input data split into train set.
    train_unscaled: Input data unprocessed split into train.
    test_unscaled: Input data unprocessed split into test.
    numerical_pipeline: Stored numerical transformation pipeline for
                        full model runs. Left as None if not a full
                        model run.
    output_folder: Path to output folder.
    Returns
    -------
    train_final: Final train data post data transformations.
    test_final: Final test data post data transformations.
    train_scaled: If transformations are not permitted, train scaled is
                  returned.
    test_scaled: If transformations are not permitted, test scaled is
                  returned.
    """

    alpha = 0.05
    issues = {
        "linearity": False,
        "normality": False,
        "multicolinearity": False,
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

    # if is_statsmodel:
    #     X_with_const = add_constant(x_test)
    #     model_for_tests = model
    # else:
    #     # scikit models, convert to stats
    #     X_with_const_train = add_constant(x_train)
    #     if is_classification:
    #         # classification
    #         model_for_tests = sm.Logit(y_train, X_with_const_train).fit(disp=0)
    #     else:
    #         # regression
    #         model_for_tests = sm.OLS(y_train, X_with_const_train).fit()
    #     residuals = model_for_tests.resid
    #     X_with_const = add_constant(x_test)

    # multicolinearity
    if train_scaled is not None and not train_scaled.empty:
        LOG.info("Checking for multicollinearity")
        vif_data = pd.DataFrame()
        vif_data["Feature"] = train_scaled.columns
        vif_data["VIF"] = [
            variance_inflation_factor(train_scaled.values, i)
            for i in range(train_scaled.shape[1])
        ]
        high_vif = vif_data[vif_data["VIF"] > 10]
        if not high_vif.empty:
            LOG.warning("High VIF variables: %s", high_vif)
            issues["multicolinearity"] = True

    if train_scaled is not None and not train_scaled.empty:
        x_with_const = add_constant(x_test)
        # Breusch-Pagan Heteroscedasticity
        LOG.info("Checking for heteroscedasticity")
        try:
            _, bp_test_p_value, _, _ = het_breuschpagan(residuals, x_with_const)
            LOG.info("Breusch-Pagan test p-value: %s", bp_test_p_value)
        except (ValueError, MissingDataError) as e:
            LOG.warning("Breusch-Pagan test failed: %s", e)
            bp_test_p_value = 1.0
        # White Test Heteroscedasticity
        try:
            _, white_test_p_value, _, _ = het_white(residuals, x_with_const)
            LOG.info("White's test p-value: %s", white_test_p_value)
        except (ValueError, MissingDataError) as e:
            LOG.warning("White's test failed: %s", e)
            white_test_p_value = 1.0
        if bp_test_p_value < alpha or white_test_p_value < alpha:
            LOG.warning("Warning: Heteroscedasticity detected.")
            issues["heteroscedasticity"] = True

    if is_linear_model and not is_classification and residuals is not None:
        LOG.info("Running tests for linear model assumptions")

        # Linearity
        LOG.info("Checking linearity")
        for col in x_train.columns:
            try:
                correlation = np.corrcoef(x_train[col], residuals)[0, 1]
                if abs(correlation) > 0.1:
                    LOG.warning("Warning: %s may not be linearly related to the target.", col)
                    issues["linearity"] = True
            except (ValueError, MissingDataError) as e:
                LOG.warning("Linearity test failed for column %s: %s", col, e)

        # Normality
        try:
            _, shapiro_p_value = shapiro(residuals)
            LOG.info("Shapiro-Wilk test p-value: %s", shapiro_p_value)
            if shapiro_p_value < alpha:
                LOG.warning("Warning: Shapiro-Wilk test suggests non-normality of residuals.")
                issues["normality"] = True
        except (ValueError, MissingDataError) as e:
            LOG.warning("Shapiro-Wilk test failed: %s", e)

    if data_classification.is_time_series and residuals is not None:
        # Autocorrelation
        try:
            dw_statistic = durbin_watson(residuals)
            LOG.info("Durbin-Watson statistic: %s", dw_statistic)
            if dw_statistic < 1.5 or dw_statistic > 2.5:
                LOG.warning("Warning: Potential autocorrelation in residuals.")
                issues["autocorrelation"] = True
        except (ValueError, MissingDataError) as e:
            LOG.warning("Durbin-Watson test failed: %s", e)

    issues_df = pd.DataFrame(list(issues.items()), columns=["test", "result"])
    if any(issues.values()):
        LOG.warning("Data issue present: %s", issues_df)

        issues_df.to_csv(output_folder / "data_issues_present.csv", index=False)
        if (
            modelling.full_transformations
            and data_classification.numerical_features
            and train_unscaled is not None
            and test_unscaled is not None
        ):

            LOG.info("Transformations applied to numerical data to fix the issues")
            train_final = transform_data(
                df=train_unscaled,
                is_test_data=False,
                numerical_pipeline=numerical_pipeline,
                output_folder=output_folder,
                data_classification=data_classification,
            )
            test_final = transform_data(
                df=test_unscaled,
                is_test_data=True,
                numerical_pipeline=numerical_pipeline,
                output_folder=output_folder,
                data_classification=data_classification,
            )
            return train_final, test_final

        LOG.warning(
            "Data issue present but no numerical features are present or full transformations have \
             not been permitted so transformations can't occur"
        )
        return train_scaled, test_scaled
    LOG.info("No data issues present.")
    return train_scaled, test_scaled


def transform_data(
    df: pd.DataFrame,
    is_test_data: bool,
    numerical_pipeline,
    output_folder: Path,
    data_classification,
):
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
    data_classification: Data classification inputs from the PredictionModelInputs
                         class. These inputs help define and outline the
                         structure of the input data. See
                         caf/brain/ml/main_models/prediction_model/ml_inputs.py
                         for available options.
    Returns
    -------
    transformed_df: Transformed data.
    """

    transformed_data = []

    original_index = df.index

    if data_classification.target_column in df.columns:
        target = df[data_classification.target_column].copy()
    else:
        target = None
    if data_classification.weight_column in df.columns:
        weight = df[data_classification.weight_column].copy()
    else:
        weight = None

    numerical_data = df[data_classification.numerical_features].copy()

    # log
    numerical_transformed = np.log1p(numerical_data)
    # numerical_transformed = numerical_data.apply(lambda x: np.log1p(x))  # log1p = log(1+x)
    # numerical_transformed = numerical_data.apply(lambda x: np.log(x + 1))

    # force fixing any issues post log transformations
    numerical_transformed = numerical_transformed.replace([np.inf, -np.inf], np.nan)
    numerical_transformed = numerical_transformed.fillna(numerical_transformed.mean())

    # scale
    if is_test_data:
        numerical_scaled = preprocess_numerical_data(
            df=numerical_transformed,
            numerical_features=data_classification.numerical_features,
            is_test_data=is_test_data,
            numerical_pipeline_train=numerical_pipeline,
            output_folder=output_folder,
        )
    else:
        numerical_scaled, _ = preprocess_numerical_data(
            df=numerical_transformed,
            numerical_features=data_classification.numerical_features,
            is_test_data=is_test_data,
            numerical_pipeline_train=numerical_pipeline,
            output_folder=output_folder,
        )

    # pca
    # pca = PCA()
    # numerical_pca = pd.DataFrame(
    #     pca.fit_transform(numerical_scaled),
    #     columns=numerical_features,
    #     index=original_index)
    #
    # transformed_data.append(numerical_pca)

    transformed_data.append(numerical_scaled)

    if data_classification.categorical_features is not None:
        categorical_data = df[data_classification.categorical_features].copy()
        transformed_data.append(categorical_data)

    if weight is not None:
        transformed_data.append(weight)

    transformed_df = pd.concat(transformed_data, axis=1)

    if target is not None:
        transformed_df[data_classification.target_column] = target
        transformed_df[data_classification.target_column] = transformed_df[
            data_classification.target_column
        ].astype(int)

    transformed_df.index = original_index

    return transformed_df

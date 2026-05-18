"""
Module for model selection _functions. Used to select and initialise a
SciKit-Learn algorithm.
"""

# Built-Ins
import logging
from pathlib import Path
from typing import Optional

# Third Party
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

# Local Imports
from caf.brain.ml._functions._ml_inputs import Models, XGBClassifierMulticlass
from caf.brain.ml._functions.process_data_functions.split_data_into_ttv import (
    sample_data,
)

LOG = logging.getLogger(__name__)


def initialise_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_folder: Path,
    model_initialised,
    classification_prediction: tuple[int, ...] | None,
    x_train_weight: Optional[pd.Series] = None,
) -> tuple[BaseEstimator, pd.Series, float | None]:
    """
    Fit the initialised model and evaluate its initial performance.

    Parameters
    ----------
    x_train: pd.DataFrame of x train values from SciKit-Learns
             train_test_split (simple_train_test_split can be used to
             generate this)
    x_test:  pd.DataFrame of x test values from SciKit-Learns
             train_test_split (simple_train_test_split can be used to
             generate this)
    y_train: pd.Series of y train values from SciKit-Learns
             train_test_split (simple_train_test_split can be used to
             generate this)
    y_test:  pd.Series of y test values from SciKit-Learns
             train_test_split (simple_train_test_split can be used to
             generate this)
    output_folder: Path to output location.
    model_initialised: Initialised SciKitLearn model.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    x_train_weight: Pandas series of weight values that correspond to
                    x_train generated in simple_train_test_split

    Returns
    -------
    model_fit: Fitted model on train_test_split test data.
    residuals: Truth values form the train_test_split against the predictions.
    mse: Mean squared error of predictions.
    """
    weight = None
    if x_train_weight is not None:
        weight = x_train_weight.to_numpy().flatten()

    model_filename = output_folder / "initial_fitted_model.pkl"
    if model_filename.exists():
        model_fit = joblib.load(model_filename)
    else:
        if weight is not None:
            model_fit = model_initialised.fit(x_train, y_train, sample_weight=weight)
        else:
            model_fit = model_initialised.fit(x_train, y_train)

        joblib.dump(model_fit, model_filename)

    y_pred = model_fit.predict(x_test)
    if classification_prediction is not None:
        if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
            y_pred = np.argmax(y_pred, axis=1)

    residuals = y_test - y_pred
    coeff_df, mse = calculate_model_coeff(
        model=model_fit,
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        residuals=residuals,
        classification_prediction=classification_prediction,
        y_pred=y_pred,
    )

    if coeff_df is not None:
        coeff_df.to_csv(output_folder / "initial_model_coefficients.csv", index=False)

    return model_fit, residuals, mse


def select_model(
    train: pd.DataFrame,
    output_folder: Path,
    target_column: str | None,
    weight_column: str | None,
    models_to_test: list[Models],
    classification_prediction: tuple[int, ...] | None,
    is_time_series: bool | None = False,
) -> BaseEstimator:
    """
    Quickly assess and select the best model from a list of candidates.

    Parameters
    ----------
    train: Processed input data split into train subset.
    output_folder: Path to output location.
    target_column: Sting column name of value to predict.
    weight_column: Optional string column value to be used as weight.
    models_to_test: List or one algorithm to use as the base of the model.
                    Available algorithms can be seen in
                    _ml_inputs.py or __info__.py.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    is_time_series: If true data is temporal is nature.

    Returns
    -------
    best_model: Best performing model initialised.
    """
    eval_df_path = Path(output_folder) / "model_algorithm_evaluation.csv"

    if eval_df_path.exists():
        LOG.info(
            "model_algorithm_evaluation.csv already exists so the best "
            "model is being selected from these results."
        )
        df = pd.read_csv(eval_df_path)
        if classification_prediction:
            best_idx = df["F1"].idxmax()
            best_model_str = str(df.loc[best_idx, "Models"])
        else:
            best_idx = df["R-squared"].idxmax()
            best_model_str = str(df.loc[best_idx, "Models"])
        enum_name = best_model_str.split(".")[1]
        best_model_enum = Models[enum_name]
        best_model = best_model_enum.get_model()

        return best_model

    if not target_column:
        raise ValueError(
            "Please provide a target column. "
            "This should be a column title passed as a string."
        )

    x = train.drop(columns=[target_column] + ([weight_column] if weight_column else []))
    y = train[target_column]
    weight = np.asarray(train[weight_column]).flatten() if weight_column else None

    acc = {}
    best_score = float("-inf")
    best_model = None

    x, y, weight = sample_data(x=x, y=y, weight=weight, is_time_series=is_time_series)

    for model_enum in models_to_test:
        # scikit
        model_instance = model_enum.get_model()
        LOG.info("Testing model: %s", model_instance)

        if isinstance(model_instance, XGBClassifierMulticlass):
            num_classes = y.nunique()
            model_instance.set_params(num_class=num_classes)

        if isinstance(model_instance, LogisticRegression):
            model_instance.set_params(max_iter=1000)

        if classification_prediction:
            scores_f1, scores_auc = score_classification(
                weight=weight, model_instance=model_instance, x=x, y=y
            )
            mean_score = scores_f1.mean()
            acc[model_enum] = {"F1": scores_f1.mean(), "AUC": -scores_auc.mean()}
        else:
            scores_r2, scores_mse = score_regression(
                weight=weight, model_instance=model_instance, x=x, y=y
            )
            mean_score = scores_r2.mean()
            acc[model_enum] = {"R-squared": scores_r2.mean(), "MSE": scores_mse.mean()}

        if mean_score > best_score:
            best_score = mean_score
            best_model = model_enum.get_model()

    LOG.info("Best model: %s", best_model)
    LOG.info("Best model score: %s", best_score)
    evaluation_df = pd.DataFrame.from_dict(acc, orient="index")
    evaluation_df.index.name = "Models"
    evaluation_df.to_csv(output_folder / "model_algorithm_evaluation.csv", index=True)

    return best_model


def score_regression(
    weight: pd.DataFrame, model_instance: Models, x: pd.DataFrame, y: pd.DataFrame
):
    """
    Score regression models using cross-validation.

    Parameters
    ----------
    weight: Pandas dataframe of weight values from the original
            train input data.
    model_instance: Initialised model.
    x: Train data split into only the explanatory variables. Target
       and weight should be removed. Any index columns should be set.
    y: Train data split into only the target. Any index columns should be set.

    Returns
    -------
    scores_r2: Series of R2 scores.
    scores_mse: Series of mean squared error scores.
    """
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    if weight is not None:
        scores_r2 = cross_val_score(
            model_instance,
            x,
            y,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            params={"sample_weight": weight},
            verbose=1,
        )
        scores_mse = -cross_val_score(
            model_instance,
            x,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            params={"sample_weight": weight},
            verbose=1,
        )
    else:
        scores_r2 = cross_val_score(
            model_instance, x, y, cv=cv, scoring="r2", n_jobs=-1, verbose=1
        )
        scores_mse = -cross_val_score(
            model_instance, x, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
        )

    return scores_r2, scores_mse


def score_classification(
    weight: pd.DataFrame, model_instance: Models, x: pd.DataFrame, y: pd.Series
):
    """
    Score classification models using cross-validation.

    Parameters
    ----------
    weight: Pandas dataframe of weight values from the original train input data.
    model_instance: Initialised model.
    x: Train data split into only the explanatory variables. Target
       and weight should be removed. Any index columns should be set.
    y: Train data split into only the target. Any index columns should be set.

    Returns
    -------
    scores_f1: Series of F1 scores.
    scores_auc: Series of AUC scores.
    """
    y_ = pd.Series(y).squeeze()
    assert isinstance(y_, pd.Series)

    if y_.nunique() > 2:
        f1 = "f1_weighted"
        roc_auc = "roc_auc_ovr_weighted"
    else:
        f1 = "f1"
        roc_auc = "roc_auc"

    if weight is not None:
        scores_f1 = cross_val_score(
            model_instance,
            x,
            y,
            cv=3,
            scoring=f1,
            n_jobs=-1,
            params={"sample_weight": weight},
            verbose=1,
        )
        scores_auc = -cross_val_score(
            model_instance,
            x,
            y,
            cv=3,
            scoring=roc_auc,
            n_jobs=-1,
            params={"sample_weight": weight},
            verbose=1,
        )
    else:
        scores_f1 = cross_val_score(
            model_instance, x, y, cv=3, scoring=f1, n_jobs=-1, verbose=1
        )
        scores_auc = -cross_val_score(
            model_instance, x, y, cv=3, scoring=roc_auc, n_jobs=-1, verbose=1
        )

    return scores_f1, scores_auc


def calculate_model_coeff(
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame | None,
    y_test: pd.Series | None,
    residuals: pd.Series | None,
    classification_prediction: tuple[int, ...] | None,
    y_pred: pd.Series,
) -> tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Calculate model coefficients and statistics.

    Parameters
    ----------
    model:
        Fitted model on train_test_split of training data.
    x_train:
        Series of train data to be used as train.
    x_test:
        Series of test data to be used as unseen test data.
    y_test:
        Series of target column inside train to be used as validation
        for predictions.
    residuals:
        Series of residual values based on x_test predictions.
    classification_prediction:
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.
    y_pred:
        Series of predicted values based on training data.

    Returns
    -------
    coeff_df:
        Dataframe of coefficient values and other relevant statistics.
    mse:
        Mean squared error of predictions.
    """
    if _is_statsmodels_model(model):
        return _extract_statsmodels_inference(model, x_test, y_test)

    if hasattr(model, "coef_"):
        if len(model.coef_.shape) == 2 and model.coef_.shape[0] > 1:
            return None, None

        return _extract_sklearn_coefficients(
            model, x_train, x_test, y_test, y_pred, residuals, classification_prediction
        )

    if hasattr(model, "feature_importances_"):
        return _extract_feat_importance(
            model, x_train, x_test, y_test, y_pred, classification_prediction
        )

    LOG.warning(
        "Model %s does not support coefficient or importance extraction",
        {type(model).__name__},
    )
    return None, None


def calculate_final_coefficients(
    model,
    test_data: pd.DataFrame,
    training_mse: float | None,
    predictions: pd.Series,
    validation_data: pd.DataFrame | None,
    target_column: str | None,
    is_classification: tuple[int, ...] | None,
    drop_vals: pd.DataFrame | None,
    cols_dropped_by_feat_select: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Calculate final model coefficients and statistics.

    Parameters
    ----------
    model:
        Fitted final model for prediction on unseen (test) data.
    test_data:
        Dataframe of final test data post feature selection.
    training_mse:
        Mean squared error of predictions based on training data.
    predictions:
        Predicted values based on the test data and set to the same index.
    validation_data:
        Validation data if available. Must pass target column if passing
        validation data otherwise validation will not be used.
    target_column:
        String column name of value to predict.
    is_classification:
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.
    drop_vals:
        Values dropped during encoding of categorical variables.
    cols_dropped_by_feat_select:
        These are the columns removed due to feature selection.

    Returns
    -------
    coeff_df:
        Dataframe of coefficient values and other relevant statistics.
    """
    if validation_data is not None and target_column is not None:
        coeff_df, error_metric = calculate_model_coeff(
            model=model,
            x_train=test_data,  # only using it for the column names
            x_test=test_data,
            y_test=validation_data[target_column],
            y_pred=predictions,
            residuals=None,
            classification_prediction=is_classification,
        )
    else:
        coeff_df, error_metric = calculate_model_coeff(
            model=model,
            x_train=test_data,
            x_test=test_data,
            y_test=None,
            y_pred=predictions,
            residuals=None,
            classification_prediction=is_classification,
        )

    if error_metric is None:
        error_metric = training_mse if training_mse is not None else None

    if coeff_df is None:
        return None

    coeff_df["Error_Metric"] = error_metric
    coeff_df["Error_Source"] = "validation" if validation_data is not None else "training"

    base_columns = list(coeff_df.columns)
    dropped_dfs = []

    if drop_vals is not None and len(drop_vals.columns) > 0:
        drop_vals_features = pd.DataFrame(
            {
                "Feature": drop_vals.columns,
                "Status": "Dropped during encoding",
            }
        )
        for col in base_columns:
            if col not in drop_vals_features.columns:
                drop_vals_features[col] = "N/A"
        dropped_dfs.append(drop_vals_features)

    if (
        cols_dropped_by_feat_select is not None
        and len(cols_dropped_by_feat_select.columns) > 0
    ):
        feat_select_features = pd.DataFrame(
            {
                "Feature": cols_dropped_by_feat_select.columns,
                "Status": "Dropped during feature selection",
            }
        )
        for col in base_columns:
            if col not in feat_select_features.columns:
                feat_select_features[col] = "N/A"
        dropped_dfs.append(feat_select_features)

    if dropped_dfs:
        coeff_df["Status"] = "Active in model"
        coeff_df = pd.concat([coeff_df] + dropped_dfs, ignore_index=True)

    return coeff_df


def _extract_sklearn_coefficients(
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame | None,
    y_test: pd.Series | None,
    y_pred: pd.Series | None,
    residuals: Optional[pd.Series] | None,
    classification_prediction: Optional[tuple[int, ...]] = None,
) -> tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Extract coefficients from scikit-learn linear models.

    No p-values or standard errors (not mathematically valid for sklearn).

    Valid for:
    - LinearRegression
    - Ridge, Lasso, ElasticNet
    - LogisticRegression (L1, L2, ElasticNet)
    - LinearSVC

    Parameters
    ----------
    model:
        Fitted model on train_test_split of training data.
    x_train:
        Series of train data to be used as train.
    x_test:
        Series of test data to be used as unseen test data.
    y_test:
        Series of target column inside train to be used as validation
        for predictions.
    residuals:
        Series of residual values based on x_test predictions.
    classification_prediction:
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.
    y_pred:
        Series of predicted values based on training data.

    Returns
    -------
    coeff_df:
        Dataframe of coefficient values and other relevant statistics.
    error_metric:
        Mean squared error of predictions.
    """
    # multinomial
    if len(model.coef_.shape) == 2 and model.coef_.shape[0] > 1:
        LOG.info("Multinomial logistic regression excluded")
        return None, None

    if hasattr(model, "intercept_"):
        intercept = np.array(model.intercept_).flatten()
        coefficients = np.array(model.coef_).flatten()
        coefficients = np.concatenate([intercept, coefficients])
        feature_names = ["intercept"] + list(x_train.columns)
    else:
        coefficients = model.coef_.flatten()
        feature_names = list(x_train.columns)

    coeff_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Abs_Coefficient": np.abs(coefficients),
        }
    )

    if isinstance(model, LogisticRegression):
        coeff_df["Odds_Ratio"] = np.exp(coefficients)
        coeff_df["Note"] = (
            "Odds ratios from sklearn LogisticRegression (no p-values available)"
        )
    else:
        coeff_df["Note"] = "Coefficients from sklearn (no statistical inference available)"

    coeff_df = coeff_df.sort_values("Abs_Coefficient", ascending=False)

    if y_test is not None:
        if classification_prediction:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_test)
                if proba.shape[1] == 2:
                    # binary
                    error_metric = log_loss(y_test, proba[:, 1])
                else:
                    # multiclass
                    error_metric = log_loss(y_test, proba)
            else:
                # linear svc
                error_metric = log_loss(y_test, y_pred, labels=np.unique(y_test))
        else:
            # regression
            error_metric = (
                np.mean(residuals**2)
                if residuals is not None
                else mean_squared_error(y_test, y_pred)
            )
        if error_metric:
            error_metric = float(error_metric)
        LOG.info("Extracted sklearn coefficients for %s features", len(coeff_df))
        return coeff_df, error_metric

    LOG.info("Extracted sklearn coefficients for %s features", len(coeff_df))
    return coeff_df, None


def _extract_feat_importance(
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame | None,
    y_test: pd.Series | None,
    y_pred: pd.Series | None,
    classification_prediction: Optional[tuple[int, ...]] = None,
) -> tuple[pd.DataFrame, Optional[float]]:
    """
    Extract feature importances from tree-based models (sklearn).

    Valid for:
    - RandomForest (Classifier/Regressor)
    - ExtraTrees (Classifier/Regressor)
    - GradientBoosting (Classifier/Regressor)
    - DecisionTree (Classifier/Regressor)
    - AdaBoost
    - Bagging

    Parameters
    ----------
    model:
        Fitted model on train_test_split of training data.
    x_train:
        Series of train data to be used as train.
    x_test:
        Series of test data to be used as unseen test data.
    y_test:
        Series of target column inside train to be used as validation
        for predictions.
    classification_prediction:
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.
    y_pred:
        Series of predicted values based on training data.

    Returns
    -------
    importance_df:
        Dataframe of importance values.
    error_metric:
        Log loss or mean squared error of predictions.
    """
    importance_df = pd.DataFrame(
        {
            "Feature": x_train.columns,
            "Importance": model.feature_importances_,
            "Importance_Type": "Gini/MDI",
        }
    ).sort_values("Importance", ascending=False)

    importance_df["Cumulative_Importance"] = importance_df["Importance"].cumsum()

    if y_test is not None:
        if classification_prediction:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_test)
                error_metric = log_loss(y_test, proba)
            else:
                error_metric = log_loss(y_test, y_pred)
        else:
            error_metric = mean_squared_error(y_test, y_pred)
        if error_metric is not None:
            error_metric = float(error_metric)
        LOG.info("Extracted feature importances for %s features", len(importance_df))
        return importance_df, error_metric
    LOG.info("Extracted feature importances for %s features", len(importance_df))
    return importance_df, None


def _is_statsmodels_model(model) -> bool:
    """
    Check if model is statsmodels model.

    Parameters
    ----------
    model:
        Fitted model on train_test_split of training data.
    Returns
    -------
    True if statsmodels algorithm.
    """
    return any(base.__module__.startswith("statsmodels") for base in model.__class__.__mro__)


def _extract_statsmodels_inference(
    model,
    x_test: pd.DataFrame | None,
    y_test: pd.Series | None,
    classification_prediction: Optional[tuple[int, ...]] = None,
) -> tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Extract full statistical inference from statsmodels.

    Valid for:
    - statsmodels.OLS (returns t-values)
    - statsmodels.Logit (returns z-values)
    - statsmodels.GLM (returns z-values)

    Parameters
    ----------
    model:
        Fitted model on train_test_split of training data.
    x_test:
        Series of test data to be used as unseen test data.
    y_test:
        Series of target column inside train to be used as validation
        for predictions.
    classification_prediction:
        List of integers that correspond to the target column. The value(s) to
        predict in a classification problem.

    Returns
    -------
    coeff_df:
        Dataframe of coefficient values and other relevant statistics.
    error_metric:
        Mean squared error or log loss of predictions.
    """
    # multinomial
    if "MNLogit" in str(model.__class__):
        LOG.info("Multinomial logit results excluded")
        return None, None

    try:
        stats_df = pd.DataFrame(
            {
                "Feature": model.params.index,
                "Coefficient": model.params.values,
                "Std_Error": model.bse.values,
                "Statistic": model.tvalues.values,  # t-value for OLS, z-value for Logit/GLM
                "P_Value": model.pvalues.values,
                "CI_Lower_95": model.conf_int()[0].values,
                "CI_Upper_95": model.conf_int()[1].values,
            }
        )

        if "Logit" in str(model.__class__):
            stats_df["Odds_Ratio"] = np.exp(stats_df["Coefficient"])
            stats_df["OR_CI_Lower_95"] = np.exp(stats_df["CI_Lower_95"])
            stats_df["OR_CI_Upper_95"] = np.exp(stats_df["CI_Upper_95"])

        if y_test is not None:
            if hasattr(model, "predict"):
                if classification_prediction:
                    # logistic regression
                    y_pred_proba = model.predict(x_test)
                    error_metric = log_loss(y_test, y_pred_proba)
                else:
                    # OLS
                    y_pred = model.predict(x_test)
                    error_metric = mean_squared_error(y_test, y_pred)
                if error_metric:
                    error_metric = float(error_metric)

                LOG.info("Extracted statsmodels inference with %s features", len(stats_df))
                return stats_df, error_metric
        else:
            LOG.info("Extracted statsmodels inference with %s features", len(stats_df))
        return stats_df, None

    except Exception as e:  # pylint: disable=broad-exception-caught
        LOG.error("Failed to extract statsmodels inference: %s", e)
        return None, None

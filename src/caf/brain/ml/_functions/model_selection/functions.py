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
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import cross_val_score

# Local Imports
from caf.brain.ml._functions._ml_inputs import Models
from caf.brain.ml._functions.process_data_functions.split_data_into_ttv import sample_data

LOG = logging.getLogger(__name__)


def initialise_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_folder: Path,
    model_initialised: BaseEstimator,
    classification_prediction: tuple[int, ...] | None,
    x_train_weight: pd.DataFrame = None,
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
    x_train_weight:  Numpy ndarray of weight values that correspond to
                     x_train generated in simple_train_test_split

    Returns
    -------
    model_fit: Fitted model on train_test_split test data.
    residuals: Truth values form the train_test_split against the predictions.
    mse: Mean squared error of predictions.
    """
    weight = None
    if x_train_weight is not None:
        weight = x_train_weight.values.flatten()

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
    is_time_series: bool = False,
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
        LOG.info("model_algorithm_evaluation.csv already exists so the best "
                 "model is being selected from these results.")
        df = pd.read_csv(eval_df_path)
        if classification_prediction:
            best_idx = df["F1"].idxmax()
            best_model_str = str(df.loc[best_idx, "Models"])
        else:
            best_idx = df["R2"].idxmax()
            best_model_str = str(df.loc[best_idx, "Models"])
        best_model_enum = [best_model_str]
        best_model = best_model_enum[0].get_model()

        return best_model

    if not target_column:
        raise ValueError(
            "Please provide a target column. "
            "This should be a column title passed as a string."
        )

    x = train.drop(columns=[target_column] + ([weight_column] if weight_column else []))
    y = train[target_column]
    weight = train[weight_column].values.flatten() if weight_column else None

    acc = {}
    best_score = float("-inf")
    best_model = None

    x, y, weight = sample_data(x=x, y=y, weight=weight, is_time_series=is_time_series)

    for model_enum in models_to_test:
        # scikit
        model_instance = model_enum.get_model()
        LOG.info("Testing model: %s", model_instance)

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
    if weight is not None:
        scores_r2 = cross_val_score(
            model_instance,
            x,
            y,
            cv=3,
            scoring="r2",
            n_jobs=-1,
            params={"sample_weight": weight},
            verbose=1,
        )
        scores_mse = -cross_val_score(
            model_instance,
            x,
            y,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            params={"sample_weight": weight},
            verbose=1,
        )
    else:
        scores_r2 = cross_val_score(
            model_instance, x, y, cv=3, scoring="r2", n_jobs=-1, verbose=1
        )
        scores_mse = -cross_val_score(
            model_instance, x, y, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
        )

    return scores_r2, scores_mse


def score_classification(
    weight: pd.DataFrame, model_instance: Models, x: pd.DataFrame, y: pd.DataFrame
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
    y_ = y.squeeze()
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
    x_test: pd.DataFrame,
    y_test: pd.Series,
    residuals: pd.Series,
    classification_prediction: tuple[int, ...] | None,
    y_pred: pd.Series,
) -> tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Calculate linear model coefficients and statistics.

    Parameters
    ----------
    model: Fitted model on train_test_split of training data.
    x_train: Series of train data to be used as train.
    x_test: Series of test data to be used as unseen test data.
    y_test: Series of target column inside train to be used as validation
            for predictions.
    residuals: Series of residual values based on x_test predictions.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    y_pred: Series of predicted values based on training data.

    Returns
    -------
    coeff_df: Dataframe of coefficient values and other relevant statistics.
    mse: Mean squared error of predictions.
    """
    if not hasattr(model, "coef_"):
        return None, None

    if len(model.coef_.shape) == 2:  # Multinomial
        return None, None

    n = x_train.shape[0]
    p = x_train.shape[1]
    dof = n - p - 1

    if classification_prediction:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_test)
            if proba.shape[1] == 2:
                # binary
                mse = log_loss(y_test, proba[:, 1])
            else:
                # multiclass
                mse = log_loss(y_test, proba)
        else:
            # linear svc
            mse = log_loss(y_test, y_pred, labels=np.unique(y_test))
    else:
        # regression
        mse = np.mean(residuals**2)

    # variance-covariance matrix
    x_with_intercept = (
        np.column_stack([np.ones(n), x_train]) if hasattr(model, "intercept_") else x_train
    )
    covariance_matrix = np.linalg.pinv(x_with_intercept.T.dot(x_with_intercept)) * mse

    std_errors = np.sqrt(np.diag(covariance_matrix))

    if hasattr(model, "intercept_"):
        intercept = np.array(model.intercept_).flatten()
        coefficients = np.array(model.coef_).flatten()
        coefficients = np.concatenate([intercept, coefficients])
    else:
        coefficients = model.coef_.flatten()

    # t-values and p-values
    t_values = coefficients / std_errors
    p_values = 2 * (1 - stats.t.cdf(abs(t_values), dof))

    feature_names = (
        ["intercept"] + list(x_train.columns)
        if hasattr(model, "intercept_")
        else list(x_train.columns)
    )

    coeff_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Std_Error": std_errors,
            "T_Value": t_values,
            "P_Value": p_values,
        }
    )

    return coeff_df, mse


def calculate_final_coefficients(
    model,
    test_data: pd.DataFrame,
    training_mse: pd.Series,
    predictions: pd.Series,
    validation_data: pd.DataFrame,
    target_column: str | None,
    is_classification: tuple[int, ...] | None,
    drop_vals: pd.DataFrame,
    cols_dropped_by_feat_select: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate final model coefficients and statistics.

    Parameters
    ----------
    model: Fitted final model for prediction on unseen (test) data.
    test_data: Dataframe of final test data post feature selection.
    training_mse: Mean squared error of predictions based on training data.
    predictions: Predicted values based on the test data and set to the
                 same index.
    validation_data: Validation data if available. Must pass target column
                     if passing validation data otherwise validation will not
                     be used.
    target_column: String column name of value to predict.
    is_classification: List of integers that correspond to the target column.
                       The value(s) to predict in a classification problem.
    drop_vals: Values dropped during encoding of categorical variables.
    cols_dropped_by_feat_select: These are the columns removed due to
                                 feature selection.

    Returns
    -------
    coeff_df: Dataframe of coefficient values and other relevant statistics.
    """
    if not hasattr(model, "coef_"):
        return None

    if validation_data is not None and target_column is not None:
        if is_classification:
            # classification
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(test_data)
                if proba.shape[1] == 2:
                    # Binary
                    mse = log_loss(validation_data[target_column], proba[:, 1])
                else:
                    # Multiclass
                    mse = log_loss(validation_data[target_column], proba)
            else:
                # LinearSVC
                mse = log_loss(
                    validation_data[target_column],
                    predictions,
                    labels=np.unique(validation_data[target_column]),
                )
            LOG.info("Using validation log loss: %s", mse)
        else:
            # regression
            mse = mean_squared_error(validation_data[target_column], predictions)
            LOG.info("Using validation MSE: %s", mse)

    else:
        mse = training_mse
        LOG.info("Using training 'log loss' if %s else 'MSE': %s", is_classification, mse)

    feature_names = list(test_data.columns)
    if hasattr(model, "intercept_"):
        coefficients = np.array(model.coef_).flatten()
        intercept = np.array(model.intercept_).flatten()
        coefficients = np.concatenate([intercept, coefficients])
        feature_names = ["intercept"] + feature_names
    else:
        coefficients = model.coef_.flatten()

    n = test_data.shape[0]
    p = test_data.shape[1]
    dof = n - p - 1

    x_with_intercept = (
        np.column_stack([np.ones(n), test_data]) if hasattr(model, "intercept_") else test_data
    )
    covariance_matrix = np.linalg.pinv(x_with_intercept.T.dot(x_with_intercept)) * mse
    std_errors = np.sqrt(np.diag(covariance_matrix))
    t_values = coefficients / std_errors
    p_values = 2 * (1 - stats.t.cdf(abs(t_values), dof))

    coeff_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Std_Error": std_errors,
            "T_Value": t_values,
            "P_Value": p_values,
            "MSE_Source": "validation" if validation_data is not None else "training",
        }
    )

    if drop_vals is not None:
        # df with extra columns
        drop_vals_features = pd.DataFrame(
            {
                "Feature": drop_vals.columns,
                "Coefficient": "N/A",
                "Std_Error": "N/A",
                "T_Value": "N/A",
                "P_Value": "N/A",
                "MSE_Source": ["dropped during encoding"] * drop_vals.shape[1],
            }
        )
    else:
        drop_vals_features = None

    if cols_dropped_by_feat_select is not None:
        feat_select_features = pd.DataFrame(
            {
                "Feature": cols_dropped_by_feat_select.columns,
                "Coefficient": "N/A",
                "Std_Error": "N/A",
                "T_Value": "N/A",
                "P_Value": "N/A",
                "MSE_Source": ["dropped during feature selection"]
                * cols_dropped_by_feat_select.shape[1],
            }
        )
    else:
        feat_select_features = None

    if (
        drop_vals_features is not None
        and len(drop_vals_features) > 0
        and feat_select_features is not None
        and len(feat_select_features) > 0
    ):
        coeff_df = pd.concat(
            [coeff_df, drop_vals_features, feat_select_features], ignore_index=True
        )
    elif drop_vals_features is not None and len(drop_vals_features) > 0:
        coeff_df = pd.concat([coeff_df, drop_vals_features], ignore_index=True)
    elif feat_select_features is not None and len(feat_select_features) > 0:
        coeff_df = pd.concat([coeff_df, feat_select_features], ignore_index=True)

    return coeff_df

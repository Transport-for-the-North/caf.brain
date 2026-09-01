"""
Created on: 1/21/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

# Third Party
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from tqdm import tqdm

# Local Imports
from caf.brain.ml._functions._ml_inputs import XGBClassifierMulticlass
from caf.brain.ml._functions.process_data_functions.split_data_into_ttv import (
    sample_data,
)

LOG = logging.getLogger(__name__)


def rf_feature_selection(
    data: pd.DataFrame,
    target_column: str | None,
    cv: str | None,
    regression_method: BaseEstimator,
    weight_column: str | None,
    classification_prediction: tuple[int, ...] | None,
    is_time_series: bool | None,
) -> pd.DataFrame:
    """
    Two stage feature selection through Random Forest importance and if
    required, a combination of algorithms.

    Parameters
    ----------
    data: Transformed input data split into training set.
    target_column: String column name of value to predict.
    cv: Cross validation method passed as a string. Any popular
        SciKitLearn methods are suitable with KFold being default if
        left as None.
    regression_method: Initialised model algorithm from Models enum class.
    weight_column: Optional string column value to be used as weight.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    is_time_series: If true then data must be time series. Time series
                    based characteristics are taken into consideration
                    during function execution.


    Returns
    -------
    dataframe_final: Training data post feature selection.

    """
    if not target_column:
        raise ValueError(
            "Please provide a target column for hyperparameter"
            " optimisation. This is a column title passed as a string."
        )

    if isinstance(regression_method, LogisticRegression):
        regression_method.set_params(max_iter=1000)
    cv = get_cv_class(cv_method=cv, splits=None, repeats=None, is_time_series=is_time_series)

    x = data.drop(columns=[target_column] + ([weight_column] if weight_column else []))
    y = data[target_column]
    weight = data[weight_column].to_numpy().flatten() if weight_column else None
    weight_df = data[weight_column] if weight_column else None

    x_sample, y_sample, weight_sample = sample_data(
        x=x, y=y, weight=weight, is_time_series=is_time_series
    )

    if isinstance(regression_method, XGBClassifierMulticlass):
        num_classes = y.nunique()
        regression_method.set_params(num_class=num_classes)

    if classification_prediction:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        n_unique_classes = len(pd.unique(y_sample))
        score_threshold = 0.6
        if n_unique_classes <= 2:
            # binary
            scoring = "accuracy"
        else:
            # multi
            scoring = "f1_weighted"
    else:
        score_threshold = -0.4
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        scoring = "neg_mean_squared_error"

    with tqdm(total=1, desc="Fitting Random Forest") as pbar:
        model.fit(x_sample, y_sample, sample_weight=weight_sample)
        pbar.update(1)

    selector = SelectFromModel(model, prefit=True)
    selected_features = x_sample.columns[selector.get_support()].tolist()

    feature_importance = pd.DataFrame(
        {"feature": x_sample.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    LOG.info("Feature Importances:")
    for _, row in feature_importance.iterrows():
        LOG.info("%s: %s", row["feature"], row["importance"])

    LOG.info("Number of features selected: %s", len(selected_features))
    if len(selected_features) == 0:
        raise ValueError(
            "No features selected by Random Forest importance threshold. \n"
            "You should evaluate your data as it is likely to be irrelevant for your prediction \n"
            "Running evaluate_data can help diagnose the problem"
        )

    params = {"sample_weight": weight_sample} if weight_sample is not None else {}
    scores = cross_val_score(
        regression_method,
        x_sample[selected_features],
        y_sample,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        params=params,
    )

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    LOG.info("Cross-validated %s score: %.4f (+/- %.4f)", scoring, mean_score, std_score)

    if classification_prediction:
        needs_intensive = mean_score < score_threshold  # Low accuracy = bad
    else:
        needs_intensive = mean_score > score_threshold

    if needs_intensive:
        LOG.warning(
            "Initial feature selection attempt was inaccurate, trying alternative method"
        )
        result_sample = feature_selection_intensive(
            x=x_sample,
            y=y_sample,
            cv=cv,
            regression_method=regression_method,
            weight=weight_sample,
            classification_prediction=classification_prediction,
        )
        intensive_features = [col for col in result_sample.columns if col != target_column]

        dataframe_final = pd.concat([x[intensive_features], y], axis=1)
        if weight_df is not None:
            dataframe_final = pd.concat([dataframe_final, weight_df], axis=1)
    else:
        dataframe_final = pd.concat([x[selected_features], y], axis=1)
        if weight_df is not None:
            dataframe_final = pd.concat([dataframe_final, weight_df], axis=1)

    return dataframe_final


def feature_selection_intensive(
    x: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    regression_method,
    weight: Optional[np.ndarray],
    classification_prediction: tuple[int, ...] | None,
) -> pd.DataFrame:
    """
    Thorough feature selection with multiple algorithms.

    Parameters
    ----------
    x: Training data split into explanatory variables only.
    y: Training data split only into the target variable.
    cv: Cross validation method passed as an initialised SciKitLearn CV
        splitter.
    regression_method: Initialised model algorithm from Models enum class.
    weight: Weight values in series form.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.

    Returns
    -------
    result: Training data post feature selection.
    """

    if classification_prediction:
        selected_features = _classification_feature_selection(x, y, weight)
        score_threshold = 0.5  # uses accuracy
    else:
        selected_features = _regression_feature_selection(x, y, weight)
        score_threshold = 0.5  # uses r2

    if len(selected_features) == 0:
        raise ValueError(
            "No features selected by intensive methods. Please \n"
            "Reevaluate your data and what you are trying to predict."
        )

    scores = []
    n_splits = getattr(cv, "n_splits", 5)
    with tqdm(total=n_splits, desc="Cross-validation") as pbar:
        for train_index, test_index in cv.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            w_train = weight[train_index] if weight is not None else None

            if w_train is not None:
                regression_method.fit(
                    x_train[selected_features], y_train, sample_weight=w_train
                )
            else:
                regression_method.fit(x_train[selected_features], y_train)
            score = regression_method.score(x_test[selected_features], y_test)
            scores.append(score)
            pbar.update(1)
    cv_score = np.mean(scores)
    LOG.info("Number of features selected: %s", len(selected_features))
    LOG.info("Cross-validated score: %s", cv_score)

    if classification_prediction:
        use_all_features = cv_score < score_threshold  # Low score = bad
    else:
        use_all_features = cv_score < score_threshold  # low score = bad

    if use_all_features:
        LOG.warning("CV score is still not optimal, feature selection is being ignored")
        result = pd.concat([x, y], axis=1)
    else:
        result = pd.concat([x[selected_features], y], axis=1)

    return result


def _classification_feature_selection(
    x: pd.DataFrame, y: pd.Series, weight: np.ndarray | None
) -> list[str]:
    """
    Feature selection algorithms for classification problems.

    Parameters
    ----------
    x: Training data split into explanatory variables only.
    y: Training data split only into the target variable.
    weight: Weight values as numpy array.

    Returns
    -------
    List of selected features based on both algorithms used.
    """
    rfe = RFE(
        estimator=LogisticRegression(random_state=42, max_iter=2000),
        n_features_to_select=min(10, x.shape[1]),
    )
    rfe.fit(x, y)
    rfe_selected = x.columns[rfe.support_].tolist()

    logit_lasso = LogisticRegression(
        penalty="l1", solver="saga", random_state=42, max_iter=2000
    )
    if weight is not None:
        logit_lasso.fit(x, y, sample_weight=weight)
    else:
        logit_lasso.fit(x, y)
    l1_selected = x.columns[abs(logit_lasso.coef_[0]) > 0].tolist()

    final_features = list(set(rfe_selected + l1_selected))
    return final_features


def _regression_feature_selection(
    x: pd.DataFrame, y: pd.Series, weight: np.ndarray | None
) -> list[str]:
    """
    Feature selection algorithms for regression problems.

    Parameters
    ----------
    x: Training data split into explanatory variables only.
    y: Training data split only into the target variable.
    weight: Weight values as numpy array.

    Returns
    -------
    List of selected features based on both algorithms used.
    """
    lasso = Lasso(alpha=0.01, random_state=42)
    if weight is not None:
        lasso.fit(x, y, sample_weight=weight)
    else:
        lasso.fit(x, y)
    lasso_selected = x.columns[abs(lasso.coef_) > 0].tolist()

    ridge = Ridge(alpha=1.0, random_state=42)
    if weight is not None:
        ridge.fit(x, y, sample_weight=weight)
    else:
        ridge.fit(x, y)
    ridge_selected = x.columns[abs(ridge.coef_) > np.mean(abs(ridge.coef_))].tolist()

    final_features = list(set(lasso_selected + ridge_selected))
    return final_features


def get_cv_class(
    cv_method: str | None,
    is_time_series: bool | None,
    splits: int | None = None,
    repeats: int | None = None,
):
    """
    Select which SciKitLearn cross validation method to use.

    Parameters
    ----------
    cv_method: Cross validation method passed as a string. Any popular
               SciKitlearn methods are suitable with KFold being default if
               left as None.
    splits: Number of splits to be used for cross validation.
    repeats: Number of repeats to be used for cross validation.
    is_time_series: If true then data must be time series. Time series
                    based characteristics are taken into consideration
                    during function execution.

    Returns
    -------
    Initialised cross validation method.

    """
    if is_time_series:
        return TimeSeriesSplit(n_splits=splits if splits else 5)

    if cv_method:
        cv_lower = cv_method.lower()
        if cv_lower == "kfold":
            return KFold(n_splits=splits if splits else 5, shuffle=True)
        if cv_lower == "stratifiedkfold":
            return StratifiedKFold(n_splits=splits if splits else 5, shuffle=True)
        if cv_lower == "repeatedkfold":
            return RepeatedKFold(
                n_splits=splits if splits else 5, n_repeats=repeats if repeats else 5
            )
        if cv_lower == "repeatedstratifiedkfold":
            return RepeatedStratifiedKFold(
                n_splits=splits if splits else 5, n_repeats=repeats if repeats else 5
            )
        if cv_lower == "timeseriessplit":
            return TimeSeriesSplit(n_splits=splits if splits else 5)

        LOG.warning("Invalid cross-validation method: %s", cv_method)
        LOG.warning("Using default cross validation method: KFold")
    else:
        LOG.info("No cross-validation method specified, using default: KFold")

    return KFold(n_splits=5, shuffle=True)


def analyse_feature_importance(
    train_transformed: pd.DataFrame,
    target_column: str | None,
    weight_column: str | None,
    output_path: Path | None,
    is_time_series: bool | None = False,
) -> pd.DataFrame:
    """
    Simple feature selection through importance and correlation metrics with
    results plotted.

    Parameters
    ----------
    train_transformed: Transformed input data split into training set.
    target_column: String column name of value to predict.
    weight_column: Optional string column value to be used as weight.
    output_path: Path to output location.
    is_time_series: If true, data is temporal in nature.

    Returns
    -------
    pd.DataFrame
        - If important features are found: the training data filtered to include
          only selected features plus the target/weight columns.
        - If no features meet importance thresholds: the original training data
          is returned unchanged.
    """
    if not output_path:
        raise ValueError("Please provide an output path")

    pd.set_option("display.float_format", lambda z: f"{z:.10f}")

    if not target_column:
        raise ValueError("A target column is required for feature selection")

    x = train_transformed.drop(
        columns=[target_column] + ([weight_column] if weight_column else [])
    )
    y = train_transformed[target_column]
    weight = train_transformed[weight_column].to_numpy().flatten() if weight_column else None

    is_classification = y.dtype == "object" or y.dtype.name == "category" or y.nunique() <= 20
    if is_classification:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    x, y, weight = sample_data(x=x, y=y, weight=weight, is_time_series=is_time_series)

    n_rows, _ = train_transformed.shape
    if n_rows > 500000:
        results_df = _analyse_feat_importance_helper(x, y, rf, -1, 3)
    else:
        results_df = _analyse_feat_importance_helper(x, y, rf, -1, 10)

    importance_metrics, filtered_data = filtering_results(
        results_df=results_df,
        target_column=target_column,
        weight_column=weight_column,
        original_data=train_transformed,
    )

    if importance_metrics.empty:
        LOG.warning(
            "All feature importance metrics are zero or near-zero. "
            "This likely indicates insufficient data or data quality issues. "
            "Returning original dataset without feature selection."
        )
        return train_transformed

    if output_path:
        create_importance_plots(results_df=importance_metrics, output_path=output_path)
        results_df.to_csv(
            os.path.join(output_path, "feature_importances.csv"), float_format="%.10f"
        )

    return filtered_data


def _analyse_feat_importance_helper(
    x: pd.DataFrame,
    y: pd.Series,
    rf: Union[RandomForestClassifier, RandomForestRegressor],
    n_jobs: int,
    n_repeats: int,
) -> pd.DataFrame:
    """
    This helper function trains the provided RandomForest model on the given
    feature matrix and target vector, then calculates:
      - Random Forest feature importances
      - Permutation importances (mean and standard deviation)
      - Correlation of each feature with the target

    Parameters
    ----------
    x: Feature matrix used for training and importance calculation.
    y: Target vector corresponding to `x`.
    rf: A scikit-learn RandomForest model instance to fit and evaluate.
    n_jobs: Number of parallel jobs to use for permutation importance.
            (Use -1 to run on all available cores).
    n_repeats: Number of random shuffles to perform for permutation importance.

    Returns
    -------
    DataFrame indexed by feature name containing:
        - importance_rf : RandomForest feature importance scores
        - importance_mean_perm : Mean permutation importance
        - importance_std_perm : Standard deviation of permutation importance
        - correlation : Absolute correlation with the target
    """
    start_time = time.time()

    rf.fit(x, y)
    results = {}

    # Random Forest importance
    importance_df = pd.DataFrame(
        {"feature": x.columns, "importance_rf": rf.feature_importances_}
    ).sort_values("importance_rf", ascending=False)
    results["random_forest_importance"] = importance_df

    # Permutation importance
    perm_importance = permutation_importance(
        rf, x, y, n_repeats=n_repeats, random_state=42, n_jobs=n_jobs
    )

    perm_importance_df = pd.DataFrame(
        {
            "feature": x.columns,
            "importance_mean_perm": perm_importance.importances_mean,
            "importance_std_perm": perm_importance.importances_std,
        }
    ).sort_values("importance_mean_perm", ascending=False)
    results["permutation_importance"] = perm_importance_df

    # Target correlations
    correlations = pd.DataFrame(
        {"feature": x.columns, "correlation": [abs(x[col].corr(y)) for col in x.columns]}
    ).sort_values("correlation", ascending=False)
    results["target_correlations"] = correlations

    results_df = pd.concat(
        [
            importance_df.set_index("feature"),
            perm_importance_df.set_index("feature")[
                ["importance_mean_perm", "importance_std_perm"]
            ],
            correlations.set_index("feature"),
        ],
        axis=1,
    )

    end_time = time.time()
    print("elapsed time:", end_time - start_time)
    return results_df


def filtering_results(
    results_df: pd.DataFrame,
    target_column: str,
    weight_column: str | None,
    original_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper function for analyse_feature_importance. Results are analysed and
    applied to input data.

    Parameters
    ----------
    results_df: Dataframe of feature selection scores.
    target_column: String column name of value to predict.
    weight_column: Optional string column value to be used as weight.
    original_data: Transformed input data split into training set.

    Returns
    -------
    important_features: Feature selection score results.
    original_data: Training data post feature selection.
    """
    important_features = results_df[
        (results_df["importance_rf"] > 0.05)
        | (results_df["correlation"] > 0.3)
        | (results_df["importance_mean_perm"] > 0.01)
    ]

    selected_features = important_features.index.tolist()
    additional_columns = []
    if target_column:
        additional_columns.append(target_column)
    if weight_column:
        additional_columns.append(weight_column)

    return important_features, original_data[selected_features + additional_columns]


def combine_results(
    train_final: pd.DataFrame,
    target_column: str | None,
    weight_column: str | None,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to apply feature selection results to training data.

    Parameters
    ----------
    train_final: Final training data post feature selection.
    target_column: String column name of value to predict.
    weight_column: Optional string column value to be used as weight.
    test: Transformed input data split into training set.

    Returns
    -------
    test_final: Final test data post feature selection.
    cols_dropped_by_feat_select: Dataframe of explanatory variables
                                 removed during feature selection.
    """
    if target_column in train_final.columns and weight_column in train_final.columns:
        df = train_final.drop(columns=[target_column, weight_column])
    elif target_column in train_final.columns:
        df = train_final.drop(columns=[target_column])
    elif weight_column in train_final.columns:
        df = train_final.drop(columns=[weight_column])
    else:
        df = train_final

    test_final = test[df.columns]

    extra_columns = test.columns.difference(df.columns)
    cols_dropped_by_feat_select = test[extra_columns]

    return test_final, cols_dropped_by_feat_select


def create_importance_plots(results_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plotting feature selection scores where applicable.

    Parameters
    ----------
    results_df: Feature selection score results.
    output_path: Path to output location.

    Returns
    -------
    None
    """
    # rf importance
    plt.figure(figsize=(12, 6))
    sb.barplot(data=results_df.reset_index().head(10), x="importance_rf", y="feature")
    plt.title("Top 10 Features by Random Forest Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "rf_importance.png"))
    plt.close()

    # Permutation importance
    plt.figure(figsize=(12, 6))
    results_plot = results_df.reset_index().head(10)
    sb.barplot(
        data=results_plot,
        x="importance_mean_perm",
        y="feature",
    )
    plt.title("Top 10 Features by Permutation Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "perm_importance.png"))
    plt.close()

    # correlations
    plt.figure(figsize=(12, 6))
    sb.barplot(data=results_df.reset_index().head(10), x="correlation", y="feature")
    plt.title("Top 10 Features by Correlation with Target")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "correlations.png"))
    plt.close()

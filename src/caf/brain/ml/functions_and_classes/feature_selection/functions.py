"""
Created on: 1/21/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import os

# Third Party
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    BaseCrossValidator,
)
from tqdm import tqdm
from pathlib import Path

LOG = logging.getLogger(__name__)


def rf_feature_selection(
    data: pd.DataFrame,
    target_column: str | None,
    cv: str | None,
    regression_method,
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
        SciKitlearn methods are suitable with KFold being default if
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
            "Please provide a target column for feature selection. \
                          This is a column title passed as a string."
        )

    if isinstance(regression_method, LogisticRegression):
        regression_method.set_params(max_iter=1000)
    cv = get_cv_class(cv_method=cv, splits=None, repeats=None, is_time_series=is_time_series)

    x = data.drop(columns=[target_column] + ([weight_column] if weight_column else []))
    y = data[target_column]
    weight = data[weight_column].values.flatten() if weight_column else None
    weight_df = data[weight_column] if weight_column else None

    if classification_prediction:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        n_unique_classes = len(pd.unique(y))
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
        model.fit(x, y, sample_weight=weight)
        pbar.update(1)

    selector = SelectFromModel(model, prefit=True)
    selected_features = x.columns[selector.get_support()].tolist()

    feature_importance = pd.DataFrame(
        {"feature": x.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    LOG.info("Feature Importances:")
    for _, row in feature_importance.iterrows():
        LOG.info("%s: %s", row["feature"], row["importance"])

    n_splits = cv.n_splits if hasattr(cv, "n_splits") else cv
    fold_scores = []
    with tqdm(total=n_splits, desc="Cross-validation") as pbar:
        for _ in range(n_splits):
            score = cross_val_score(
                regression_method,
                x[selected_features],
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0,
            )[0]
            fold_scores.append(score)
            pbar.update(1)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    LOG.info("Number of features selected: %s", len(selected_features))
    LOG.info("Cross-validated ROC AUC score: %s %s", mean_score, std_score)

    needs_intensive = (
        (mean_score < score_threshold)
        if classification_prediction
        else (mean_score > score_threshold)
    )

    if needs_intensive:
        LOG.warning(
            "Initial feature selection attempt was inaccurate, trying alternative method"
        )
        dataframe_final = feature_selection_intensive(
            x=x,
            y=y,
            cv=cv,
            regression_method=regression_method,
            weight=weight,
            weight_df=weight_df,
            classification_prediction=classification_prediction,
        )
    else:
        dataframe_final = pd.concat([x[selected_features], y], axis=1)
        dataframe_final = pd.concat([dataframe_final, weight_df], axis=1)
    return dataframe_final


def feature_selection_intensive(
    x: pd.DataFrame,
    y: pd.DataFrame,
    cv: BaseCrossValidator,
    regression_method,
    weight: pd.Series,
    weight_df: pd.DataFrame,
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
    weight_df: Weight values in a dataframe.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.

    Returns
    -------
    result: Training data post feature selection.
    """
    original_index = x.index

    if classification_prediction:
        selected_features = _classification_feature_selection(x, y, weight)
    else:
        selected_features = _regression_feature_selection(x, y, weight)

    scores = []
    with tqdm(total=cv.n_splits, desc="Cross-validation") as pbar:
        for train_index, test_index in cv.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            regression_method.fit(x_train[selected_features], y_train)
            score = regression_method.score(x_test[selected_features], y_test)
            scores.append(score)
            pbar.update(1)

    LOG.info("Number of features selected: %s", len(selected_features))
    LOG.info("Cross-validated score: %s", np.mean(scores))

    cv_score = np.mean(scores)
    if cv_score < 0.5:
        LOG.warning("CV score is still not optimal, feature selection is being ignored")
        result = pd.concat([x, y], axis=1)
        if weight_df is not None:
            result = pd.concat([result, weight_df], axis=1)
        return result.set_index(original_index)

    result = pd.concat([x[selected_features], y], axis=1)
    if weight_df is not None:
        result = pd.concat([result, weight_df], axis=1)
    return result.set_index(original_index)


def _classification_feature_selection(
    x: pd.DataFrame, y: pd.DataFrame, weight: pd.Series
) -> list:
    """
    Feature selection algorithms for classification problems.

    Parameters
    ----------
    x: Training data split into explanatory variables only.
    y: Training data split only into the target variable.
    weight: Weight values in series form.

    Returns
    -------
    List of selected features based on both algorithms used.
    """
    rfe = RFE(
        estimator=LogisticRegression(random_state=42, max_iter=2000), n_features_to_select=10
    )
    rfe.fit(x, y, sample_weight=weight)
    rfe_selected = x.columns[rfe.support_].tolist()

    logit_lasso = LogisticRegression(penalty="l1", solver="saga", random_state=42)
    logit_lasso.fit(x, y, sample_weight=weight)
    l1_selected = x.columns[abs(logit_lasso.coef_[0]) > 0].tolist()

    final_features = list(set(rfe_selected + l1_selected))
    return final_features


def _regression_feature_selection(x: pd.DataFrame, y: pd.DataFrame, weight: pd.Series) -> list:
    """
    Feature selection algorithms for regression problems.

    Parameters
    ----------
    x: Training data split into explanatory variables only.
    y: Training data split only into the target variable.
    weight: Weight values in series form.

    Returns
    -------
    List of selected features based on both algorithms used.
    """
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(x, y, sample_weight=weight)
    lasso_selected = x.columns[abs(lasso.coef_) > 0].tolist()

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x, y, sample_weight=weight)
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
    if is_time_series is True:
        return TimeSeriesSplit(n_splits=splits if splits else 5)
    if cv_method:
        if cv_method.lower() == "kfold":
            return KFold(n_splits=splits if splits else 5, shuffle=True)
        if cv_method.lower() == "stratifiedkfold":
            return StratifiedKFold(n_splits=splits if splits else 5, shuffle=True)
        if cv_method.lower() == "repeatedkfold":
            return RepeatedKFold(
                n_splits=splits if splits else 5, n_repeats=repeats if repeats else 5
            )
        if cv_method.lower() == "repeatedstratifiedkfold":
            return RepeatedStratifiedKFold(
                n_splits=splits if splits else 5, n_repeats=repeats if repeats else 5
            )
        if cv_method.lower() == "timeseriessplit":
            return TimeSeriesSplit(n_splits=splits if splits else 5)
    else:
        LOG.error("Invalid cross-validation method: %s", cv_method)
        raise ValueError(f"Invalid cross-validation method: {cv_method}")

    return KFold(n_splits=5, shuffle=True)


def analyse_feature_importance(
    train_transformed: pd.DataFrame,
    target_column: str | None,
    weight_column: str | None,
    output_path: Path,
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

    Returns
    -------
    filtered_data: Training data post feature selection.
    """
    # pd.set_option("display.float_format", lambda x: "%.10f" % x)
    pd.set_option("display.float_format", lambda z: f"{z:.10f}")

    if not target_column:
        raise ValueError("A target column is required for feature selection")

    x = train_transformed.drop(
        columns=[target_column] + ([weight_column] if weight_column else [])
    )
    y = train_transformed[target_column]

    is_classification = len(np.unique(train_transformed[target_column])) <= 2
    if is_classification:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    rf.fit(x, y)
    results = {}

    # Random Forest importance
    importance_df = pd.DataFrame(
        {"feature": x.columns, "importance_rf": rf.feature_importances_}
    ).sort_values("importance_rf", ascending=False)
    results["random_forest_importance"] = importance_df

    # Permutation importance
    perm_importance = permutation_importance(
        rf, x, y, n_repeats=10, random_state=42, n_jobs=-1
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


def filtering_results(
    results_df: pd.DataFrame,
    target_column: str,
    weight_column: pd.DataFrame,
    original_data: pd.DataFrame,
) -> pd.DataFrame:
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
) -> pd.DataFrame:
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
        # xerr=results_plot['importance_std_perm']
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

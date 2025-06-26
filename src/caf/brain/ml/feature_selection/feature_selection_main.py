# -*- coding: utf-8 -*-
"""
Created on: 1/21/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from pathlib import Path
import pandas as pd
from caf.brain.ml.feature_selection.feature_selection_functions import (
    rf_feature_selection,
    combine_results,
    analyse_feature_importance,
)
import logging

LOG = logging.getLogger(__name__)


def main_feature_selection(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    cv: str,
    regression_method,
    weight_column: str,
    classification_prediction: tuple[int, ...],
    output: Path,
    skip_feature_selection: bool,
    intensive_feature_selection: bool,
    is_time_series: bool,
):
    """
    Main feature selection function.

    :param train: Transformed input data split into training set.
    :param test: Transformed input data split into training set.
    :param target_column: String column name of value to predict.
    :param cv: Cross validation method passed as a string. Any popular
               SciKitlearn methods are suitable with KFold being default if
               left as None.
    :param regression_method: Initialised model algorithm from Models enum class.
    :param weight_column: Optional string column value to be used as weight.
    :param classification_prediction: List of integers that correspond to the
                                      target column. The value(s) to predict
                                      in a classification problem.
    :param output: Path to output location.
    :param skip_feature_selection: If true then feature selection is skipped.
    :param intensive_feature_selection: If True then more invasive feature
                                        selection is conducted.
    :param is_time_series: If true then data must be time series. Time series
                           based characteristics are taken into consideration
                           during function execution.

    :return:
        train_final: Dataframe of final training data post feature selection.
        test_final: Dataframe of final test data post feature selection.
        cols_dropped_by_feat_select: These are the columns removed due to
                                     feature selection.
    """
    if skip_feature_selection:
        return train, test, None

    if intensive_feature_selection:
        # eval no. samples
        n_features = (
            len(train.columns)
            - (1 if target_column in train.columns else 0)
            - (1 if weight_column else 0)
        )
        n_samples = len(train)

        min_samples_per_feature = 15
        required_samples = n_features * min_samples_per_feature

        if n_samples < required_samples:
            LOG.warning(
                f"Insufficient data for intensive feature selection. "
                f"Falling back to basic feature importance analysis."
            )
            train_final = analyse_feature_importance(
                train_transformed=train,
                target_column=target_column,
                weight_column=weight_column,
                output_path=output,
            )

            test_final, cols_dropped_by_feat_select = combine_results(
                train_final=train_final,
                target_column=target_column,
                weight_column=weight_column,
                test=test,
            )

            return train_final, test_final, cols_dropped_by_feat_select

        else:
            train_final = rf_feature_selection(
                data=train,
                target_column=target_column,
                cv=cv,
                regression_method=regression_method,
                weight_column=weight_column,
                classification_prediction=classification_prediction,
                is_time_series=is_time_series,
            )

            test_final, cols_dropped_by_feat_select = combine_results(
                train_final=train_final,
                target_column=target_column,
                weight_column=weight_column,
                test=test,
            )

            return train_final, test_final, cols_dropped_by_feat_select

    else:
        train_final = analyse_feature_importance(
            train_transformed=train,
            target_column=target_column,
            weight_column=weight_column,
            output_path=output,
        )

        test_final, cols_dropped_by_feat_select = combine_results(
            train_final=train_final,
            target_column=target_column,
            weight_column=weight_column,
            test=test,
        )

        return train_final, test_final, cols_dropped_by_feat_select

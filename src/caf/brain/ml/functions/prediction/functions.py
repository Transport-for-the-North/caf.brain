"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import os
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.svm import LinearSVC

# Local Imports
from caf.brain.ml.functions.model_selection.functions import (
    calculate_final_coefficients,
)

LOG = logging.getLogger(__name__)


def prediction(
    model,
    test: pd.DataFrame,
    target_column: str | None,
    output_folder: Path,
    validation: pd.DataFrame,
    weight_column: str | None,
    classification_prediction: tuple[int, ...] | None,
    mse: pd.Series,
    drop_vals: pd.DataFrame,
    cols_dropped_by_feat_select: pd.DataFrame,
):
    """
    Generate predictions and final model coefficients, saving results to disk.

    Parameters
    ----------
    model: Fitted final model for prediction on unseen (test) data.
    test: Dataframe of final test data post feature selection.
    target_column: String column name of value to predict.
    output_folder: Path to output location.
    validation: Validation data if available.
    weight_column: Optional string column value to be used as weight.
    classification_prediction: List of integers that correspond to the
                               target column. The value(s) to predict
                               in a classification problem.
    mse: Mean squared error or None. Dependency on if the algorithm selected
         has coefficient values.
    drop_vals: Values dropped during encoding of categorical variables.
    cols_dropped_by_feat_select: These are the columns removed due to feature
                                 selection.

    Returns
    -------
    predictions: Predicted values based on the test data and set to the same
                 index.
    """
    if validation and not target_column:
        raise ValueError(
            "Please provide a target column for prediction as you \
                          have passed a validation set of data. The target column \
                          if a string of the column title."
        )

    if target_column in test.columns:
        test = test.drop(columns=target_column)

    if weight_column in test.columns:
        weight = test[weight_column].values.flatten()
    else:
        weight = None

    if classification_prediction is not None:
        if validation is not None:
            if isinstance(model, LinearSVC):
                pred_classes = model.predict(test)
                y_true = validation[target_column].values
                accuracy = accuracy_score(y_true, pred_classes, sample_weight=weight)
            else:
                pred_probs = model.predict_proba(test)
                pred_classes = model.classes_[np.argmax(pred_probs, axis=1)]
                y_true = validation[target_column].values
                accuracy = accuracy_score(y_true, pred_classes, sample_weight=weight)

            LOG.info("Accuracy: %s", accuracy)
            accuracy_df = pd.DataFrame({"accuracy": [accuracy]})
            accuracy_df.to_csv(os.path.join(output_folder, "model_performance.csv"))
            predictions = pred_classes
        else:
            if isinstance(model, LinearSVC):
                pred_classes = model.predict(test)
            else:
                pred_probs = model.predict_proba(test)
                pred_classes = model.classes_[np.argmax(pred_probs, axis=1)]
            predictions = pred_classes
    else:
        predictions = model.predict(test)
        if validation is not None:
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
        drop_vals=drop_vals,
        cols_dropped_by_feat_select=cols_dropped_by_feat_select,
    )
    if coeff_df is not None:
        coeff_df.to_csv(
            os.path.join(output_folder, "final_model_coefficients.csv"), index=False
        )

    final_predictions = pd.DataFrame(
        {"predicted_target_column": predictions}, index=test.index
    )
    final_predictions.to_csv(os.path.join(output_folder, "final_predictions.csv"))

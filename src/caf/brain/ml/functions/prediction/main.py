"""
Created on: 1/24/2025
Original author: Adil Zaheer
"""

# Built-Ins
from pathlib import Path

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml.functions.prediction.functions import prediction


def main_prediction(
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
) -> None:
    """
    Main prediction function.

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
    cols_dropped_by_feat_select: These are the columns removed due to
                                 feature selection.

    Returns
    -------
    None
    """

    prediction(
        model=model,
        test=test,
        target_column=target_column,
        output_folder=output_folder,
        validation=validation,
        weight_column=weight_column,
        classification_prediction=classification_prediction,
        mse=mse,
        drop_vals=drop_vals,
        cols_dropped_by_feat_select=cols_dropped_by_feat_select,
    )

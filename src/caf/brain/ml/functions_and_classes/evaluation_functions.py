# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 11/20/2024
Original author: Adil Zaheer
"""
# Built-Ins
import os

# Third Party
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# todo move functions over make them work


def simple_eval_model(validation_df, y_pred, target_column, output_folder):

    y_truth = validation_df[target_column]

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_truth, y_pred, average="weighted"
    )
    metrics_dict = {"Precision": precision, "Recall": recall, "F1-score": fscore}

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(os.path.join(output_folder, "model_evaluation_metrics.csv"), index=False)

    return

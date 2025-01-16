# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


# todo move and refine prediction funcs


def final_prediction(model, data, target_column, output_folder, validation, binary_prediction):
    print('Prediction beginning')
    print(data.columns)
    print(data)
    if target_column in data.columns:
        data = data.drop(columns=target_column)

    accuracy = None
    pred_classes = None
    if binary_prediction == '0vs1':
        if isinstance(model, LinearSVC):
            pred_classes = model.predict(data)
            y_true = validation[target_column].values
            accuracy = accuracy_score(y_true, pred_classes)

        else:
            pred_probs = model.predict_proba(data)
            pred_classes = np.argmax(pred_probs, axis=1)
            y_true = validation[target_column].values
            accuracy = accuracy_score(y_true, pred_classes)

    if binary_prediction == '1vs2':
        if isinstance(model, LinearSVC):
            pred_classes = model.predict(data)
            y_true = validation[target_column].values
            accuracy = accuracy_score(y_true, pred_classes)

        else:
            pred_probs = model.predict_proba(data)
            pred_classes = np.argmax(pred_probs, axis=1) + 1
            y_true = validation[target_column].values
            accuracy = accuracy_score(y_true, pred_classes)


    print(f'Accuracy: {accuracy}')
    accuracy_df = pd.DataFrame({'accuracy': [accuracy]})
    accuracy_df.to_csv(os.path.join(output_folder, 'accuracy.csv'))

    final_predictions = pd.DataFrame({'predicted_target_column': pred_classes}, index=data.index)
    final_predictions.to_csv(os.path.join(output_folder, 'final_predictions_probability.csv'))
    print('Prediction finished')
    return pred_classes


def final_prediction_no_validation(model, data, target_column, output_folder, validation, binary_prediction):
    print('Prediction beginning')
    print(data.columns)
    print(data)
    if target_column in data.columns:
        data = data.drop(columns=target_column)

    pred_classes = None
    if binary_prediction == '0vs1':
        if isinstance(model, LinearSVC):
            pred_classes = model.predict(data)
        else:
            pred_probs = model.predict_proba(data)
            pred_classes = np.argmax(pred_probs, axis=1)

    if binary_prediction == '1vs2':
        if isinstance(model, LinearSVC):
            pred_classes = model.predict(data)
        else:
            pred_probs = model.predict_proba(data)
            pred_classes = np.argmax(pred_probs, axis=1) + 1


    final_predictions = pd.DataFrame({'predicted_target_column': pred_classes}, index=data.index)
    final_predictions.to_csv(os.path.join(output_folder, 'final_predictions_probability.csv'))
    print('Prediction finished')
    return pred_classes

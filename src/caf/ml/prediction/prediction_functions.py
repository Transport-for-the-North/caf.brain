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
from sklearn.metrics import r2_score, mean_squared_error
from caf.ml.model_selection.model_selection_functions import calculate_final_coefficients


def prediction(model,
               test,
               target_column,
               output_folder,
               validation,
               weight_column,
               classification_prediction,
               mse):
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
                # unique_classes = sorted(validation[target_column].unique())
                # pred_classes = unique_classes[np.argmax(pred_probs, axis=1)]
                pred_classes = model.classes_[np.argmax(pred_probs, axis=1)]
                y_true = validation[target_column].values
                accuracy = accuracy_score(y_true, pred_classes, sample_weight=weight)

            print(f'Accuracy: {accuracy}')
            accuracy_df = pd.DataFrame({'accuracy': [accuracy]})
            accuracy_df.to_csv(os.path.join(output_folder, 'model_performance.csv'))
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
            mse = mean_squared_error(validation[target_column], predictions, sample_weight=weight)
            metrics_df = pd.DataFrame({'r2': [r2], 'mse': [mse]})
            metrics_df.to_csv(os.path.join(output_folder, 'model_performance.csv'))

    # if hasattr(model, 'coef_'):
    #     coefficients = model.coef_
    #
    #     coefficients = np.squeeze(coefficients)
    #
    #     if coefficients.ndim == 1:
    #         coeff_df = pd.DataFrame({
    #             'Feature': test.columns,
    #             'Coefficient': coefficients
    #         })
    #     else:
    #         coeff_df = pd.DataFrame(coefficients.T, columns=test.columns)
    #         coeff_df.insert(0, 'Feature', test.columns)

    coeff_df = calculate_final_coefficients(model=model,
                                            test_data=test,
                                            training_mse=mse,
                                            predictions=predictions,
                                            validation_data=validation,
                                            target_column=target_column,
                                            is_classification=classification_prediction)
    if coeff_df is not None:
        coeff_df.to_csv(os.path.join(output_folder, 'final_model_coefficients.csv'), index=False)

    final_predictions = pd.DataFrame({'predicted_target_column': predictions}, index=test.index)
    final_predictions.to_csv(os.path.join(output_folder, 'final_predictions.csv'))
    return predictions

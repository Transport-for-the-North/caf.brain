# -*- coding: utf-8 -*-
"""
Created on: 12/28/2023
Original author: Adil Zaheer
"""
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import pandas as pd
from statsmodels.stats.diagnostic import linear_rainbow
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from sklearn.metrics import accuracy_score

from caf.ml.inputs.cafml_inputs import Models


def assumption_analysis(model_type, model, x_train, x_test, y_train, y_test):
    alpha = 0.05  # Standard alpha level for significance

    if is_regressor(model_type):
        # Rainbow Test for Linearity
        rainbow_statistic, rainbow_p_value = linear_rainbow(model)
        print(f"Rainbow test p-value: {rainbow_p_value}")
        if rainbow_p_value < alpha:
            print("Warning: Rainbow test suggests non-linearity.")

        # Shapiro-Wilk Test for Normality
        shapiro_statistic, shapiro_p_value = shapiro(model.resid)
        print(f"Shapiro-Wilk test p-value: {shapiro_p_value}")
        if shapiro_p_value < alpha:
            print("Warning: Shapiro-Wilk test suggests non-normality of residuals.")

        # Breusch-Pagan Test for Heteroscedasticity
        bp_test_statistic, bp_test_p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)
        print(f"Breusch-Pagan test p-value: {bp_test_p_value}")
        if bp_test_p_value < alpha:
            print("Warning: Breusch-Pagan test suggests heteroscedasticity.")

        # White's Test for Heteroscedasticity
        white_test_statistic, white_test_p_value, _, _ = het_white(model.resid, model.model.exog)
        print(f"White's test p-value: {white_test_p_value}")
        if white_test_p_value < alpha:
            print("Warning: White's test suggests heteroscedasticity.")

    elif isinstance(model_type, (Models.DECISION_TREE, Models.RANDOM_FOREST)):

        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"Accuracy on training set: {train_accuracy}")
        print(f"Accuracy on testing set: {test_accuracy}")

        # Check for overfitting
        if train_accuracy > test_accuracy:
            print("Warning: The model may be overfitting as training accuracy is higher than testing accuracy.")
        else:
            print("The model does not show clear signs of overfitting.")

def is_regressor(model_type):
    # Check if the model type is a regression model
    return model_type in [Models.LINEAR_REGRESSION, Models.RIDGE, Models.LASSO, Models.ELASTICNET]


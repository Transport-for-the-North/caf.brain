# -*- coding: utf-8 -*-
"""
Created on: 1/30/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# TODO make stats model pipeline, decide on data flow.
# TODO all functions written in statsmodels_functions_backlog
# TODO data analysis will need to be done first, strict data processing required for statsmodels


def main_stats_model(model_choice, train, target_column, weight_column):
    x = train.drop(columns=[target_column])
    x = sm.add_constant(x)
    y = train[target_column]

    model_enum = model_choice[0]
    model_class, params = model_enum.value
    model_initialised = model_class(train[target_column], x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

    weight = None
    if weight_column in train.columns:
        weight_df = x_train[weight_column]
        weight = weight_df.values.flatten()
        x_train = x_train.drop(columns=weight_column)
        x_test = x_test.drop(columns=weight_column)

    return

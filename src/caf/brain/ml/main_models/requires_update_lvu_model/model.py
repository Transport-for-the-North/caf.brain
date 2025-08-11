# -*- coding: utf-8 -*- MODEL
"""
Created on: 04/09/2023
Updated on: 08/09/2023

Original author: Isaac Scott
Last update made by: Isaac Scott
Other updates made by: Adil Zaheer

File purpose: Model.py feeds into the land value uplift model. It contains
cross validation and model prediction functions_to_be_processed.

"""
# Third Party
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm


def cross_validation(x, y, folds, model):
    """
    Standard cross validation function that uses repeated k fold and inputted
    model.
    :param x: The input features
    :param y: The target variable
    :param folds: Number of cross validation folds
    :param model: The model that is used to conduct the cross validation
    :return: The mean R2 and mean of the mean squared error across all folds
    is returned. These show how well the model (cross validation) is performing.
    The predicted y values are also returned. These y values are not forecasted
    y's but instead historical predictions.
    """
    mse_scores = []
    y_pred_list = []
    r_squared = []
    importances = []

    joined = x.join(y.to_frame())
    x_fixed = np.array(joined[x.columns])
    y_fixed = np.array(joined.drop(x.columns, axis=1))

    kf = RepeatedKFold(n_splits=folds)
    for train_index, test_index in tqdm(kf.split(x_fixed), desc="Model is running"):
        x_train, x_test = x_fixed[train_index], x_fixed[test_index]
        y_train, y_test = y_fixed[train_index], y_fixed[test_index]
        model.fit(x_train, y_train.ravel())
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r_squared.append(r2)
        y_pred_list.extend(y_pred)

    mean_mse = np.mean(mse_scores)
    mean_actual_r_squared = np.mean(r_squared)

    return mean_mse, y_pred_list, mean_actual_r_squared

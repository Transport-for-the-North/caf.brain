"""
Created on: 1/11/2023
Updated on:

Original author: Adil Zaheer
Last update made by: Adil Zaheer
Other updates made by: Adil Zaheer

File purpose: Create a generic cross validation function for use across models

"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from caf.ml.functions.feature_selection import get_cv_class


def cross_validation(
    df: pd.DataFrame,
    splits: int = 5,
    cv_method=None,
    model_type=None,
    repeats: int = 1,
    target_column,
):

    features = df.drop(target_column, axis=1)
    target = df[target_column]

    mse_scores = []
    y_pred_list = []
    r_squared = []

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    roc_auc_list = []

    if cv_method is None:
        cv_class = KFold
    else:
        cv_class = get_cv_class(cv_method, splits, repeats)

    if "repeated" in str(cv_method):
        cv = cv_class(n_splits=splits, n_repeats=repeats)
    else
        cv = cv_class(n_splits=splits)

    for train_index, test_index in tqdm(
        cv.split(features, target), desc="Cross validation is running"
    ):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        y_train = y_train.values.ravel() if isinstance(y_train, pd.Series) else y_train
        y_test = y_test.values.ravel() if isinstance(y_test, pd.Series) else y_test

        print("Model type:", model_type)
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)

        model_type.fit(x_train, y_train)

        if "classifier" in model_type:
            y_pred = model_type.predict(x_test)
            y_pred_list.extend(y_pred)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            f1_score = metrics.f1_score(y_test, y_pred)
            roc_auc = metrics.roc_auc_score(y_test, y_pred)

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
            roc_auc_list.append(roc_auc)
        else:
            y_pred = model_type.predict(x_test)
            y_pred_list.extend(y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mse_scores.append(mse)
            r_squared.append(r2)

    if "classifier" in model_type:
        # Calculate and return average classifier metrics
        return {
            "accuracy": np.mean(accuracy_list),
            "precision": np.mean(precision_list),
            "recall": np.mean(recall_list),
            "f1_score": np.mean(f1_score_list),
            "roc_auc": np.mean(roc_auc_list),
        }
    else:
        # Calculate and return average regression metrics
        return np.mean(mse_scores), y_pred_list, np.mean(r_squared)

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

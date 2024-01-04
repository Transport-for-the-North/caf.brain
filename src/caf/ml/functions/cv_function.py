"""
Created on: 1/11/2023
Updated on:

Original author: Adil Zaheer
Last update made by: Adil Zaheer
Other updates made by: Adil Zaheer

File purpose: Create a generic cross validation function for use across models

"""
# general imports
import pandas as pd
import numpy as np
from tqdm import tqdm

# model imports
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso, Ridge

# cv imports
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics


def cross_validation(
    final_data: pd.DataFrame,
    splits: int = 5,
    cv_method=None,
    model_type=None,
    repeats: int = 1,
    y_column=None,
):
    features = final_data.drop(columns=[y_column])
    target = final_data[y_column]

    mse_scores = []
    y_pred_list = []
    r_squared = []

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    roc_auc_list = []

    model_classes = {
        "linear_regression": LinearRegression,
        "decision_tree": DecisionTreeRegressor,
        "decision_tree_classifier": DecisionTreeClassifier,
        "random_forest": RandomForestRegressor,
        "random_forest_classifier": RandomForestClassifier,
        "svm": SVR,
        "svc": SVC,
        "knr": KNeighborsRegressor,
        "knc": KNeighborsClassifier,
        "neural_network": MLPRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "gradient_boosting_classifier": GradientBoostingClassifier,
        "naive_bayes": GaussianNB,
        "lasso": Lasso,
        "ridge": Ridge,
    }

    if model_type not in model_classes:
        raise ValueError("Invalid model type specified")
    model = model_classes[model_type]

    cv_method = KFold if cv_method is None else cv_method
    cv_class = cv_method
    if "repeated" in str(cv_method):
        cv = cv_class(n_splits=splits, n_repeats=repeats)
    else:
        cv = cv_class(n_splits=splits)

    for train_index, test_index in tqdm(
        cv.split(features, target), desc="Model is running"
    ):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        y_train = y_train.values.ravel() if isinstance(y_train, pd.Series) else y_train
        y_test = y_test.values.ravel() if isinstance(y_test, pd.Series) else y_test

        print("Model type:", model_type)
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)

        model.fit(x_train, y_train)

        if "classifier" in model_type:
            y_pred = model.predict(x_test)
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
            y_pred = model.predict(x_test)
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


def main(cv_method, model_type, y_column, final_data):
    final_data = pd.read_csv(final_data)
    final_data = final_data.drop(columns="geo_code").reset_index()
    test = cross_validation(final_data, model_type=model_type, y_column=y_column)
    print(test)

    return


if __name__ == "__main__":
    cv_method = None
    model_type = "lasso"
    y_column = "No cars or vans in household"
    final_data = r"E:\caf.ml\data_process_function\test_data\census_data.csv"
    main(cv_method, model_type, y_column, final_data)

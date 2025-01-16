# -*- coding: utf-8 -*-
"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""
import abc
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from pathlib import Path
from typing import Optional, List, Union, Any
import numpy as np
from caf.toolkit import BaseConfig
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class run_file_inputs(BaseConfig):
    # # # INPUT/OUTPUT PATHS # # #
    file_path: Optional[Path] = None
    folder_path: Optional[Path] = None
    output_path: Optional[Path] = None
    validation_path: Optional[Path] = None

    # # # PROCESSING INPUT DATA # # #
    target_column: Optional[str] = None
    custom_index: Optional[List[str]] = None
    column_name_to_drop_rows: Optional[List[str]] = None
    value_in_row: Optional[List[Union[str, int, float]]] = None

    # # # TRANSFORMING DATA # # #
    full_transformations: Optional[bool] = False
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None
    weight_column: Optional[str] = None
    binary_prediction: Optional[str] = None
    time_series_split: Optional[str] = None

    # # # MODELLING # # #
    model_choice: Any  # TODO should be abc method but not compatible with baseconfig?


class ModelStorage:
    def __init__(self):
        self.gb = GradientBoostingClassifier()
        self.rf = RandomForestClassifier()
        self.dt = DecisionTreeClassifier()
        self.svm = OneVsRestClassifier(LinearSVC())
        self.svm_binary = LinearSVC()
        self.logit_l1 = LogisticRegression(penalty='l1', solver='liblinear')
        self.logit_l2 = LogisticRegression(penalty='l2')
        self.logit_elastic_net = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5
        )
        self.logit_multinomial = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs'
        )


class ParamGridStorage:
    def __init__(self):
        self.gb_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7]
        }
        self.rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        self.dt_params = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.svm_params = {
            'estimator__C': [0.1, 1, 10],
            'estimator__loss': ['hinge', 'squared_hinge'],
        }
        self.svm_binary_params = {
            'C': [0.1, 1, 10],
            'loss': ['hinge', 'squared_hinge'],
            'penalty': ['l2'],
        }
        self.logit_l1_params = {"C": [1.0, 0.1, 0.01, 0.001]}
        self.logit_l2_params = {"C": [1.0, 0.1, 0.01, 0.001]}
        self.logit_elastic_net_params = {"C": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}
        self.logit_multinomial_params = {"C": np.arange(0.1, 10, 0.1).tolist(), "solver": ["lbfgs", "newton-cg", "sag", "saga"]}


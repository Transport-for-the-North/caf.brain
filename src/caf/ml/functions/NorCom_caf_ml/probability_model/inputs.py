# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 10/17/2024
Original author: Adil Zaheer
"""
from pathlib import Path
from caf.toolkit import BaseConfig
from typing import Optional, List, Any, Union
import enum
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


class NorCom_probability_model_inputs_cafml(BaseConfig):
    classified_build: Optional[Path] = None
    output_folder: Optional[Path] = None
    target_column: Optional[str] = None
    index_columns: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None
    weight_column: Optional[str] = None
    training_year: Optional[str] = None
    column_name_to_drop_rows: Optional[List[str]] = None
    value_in_row: Optional[List[Union[str, int, float]]] = None
    stats_model: Optional[str] = None
    sklearn_model: Optional[str] = None
    svm_adaptation: Optional[str] = None
    cafml_version: Optional[str] = None
    improve_data: Optional[str] = None
    model_to_use: Optional[str] = None
    binary_prediction: Optional[str] = None


class ModelStorage:
    def __init__(self):
        self.gb = GradientBoostingClassifier()
        self.rf = RandomForestClassifier()
        self.dt = DecisionTreeClassifier()
        self.svm = OneVsRestClassifier(LinearSVC())
        self.svm_binary = LinearSVC()


class ParamGridStorage:
    def __init__(self):
        self.gb_params = {
            'n_estimators': [100, 200, 300],
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

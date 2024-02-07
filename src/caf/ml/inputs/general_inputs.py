# -*- coding: utf-8 -*-
"""
input classes for ml models
"""
import abc
# Built-Ins
from pathlib import Path
# Third Party
from caf.toolkit import BaseConfig
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
from typing import Optional, List, Any
import enum
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


class LvuInputs(BaseConfig):
    folder_path: Path
    y_data_path: Path
    custom_column_names: list[str]
    folds: int
    target_year: int


class CarInputs(BaseConfig):
    path_2011: Path
    path_2021: Path
    method: Any  # abc.ABCMeta not supported by caf.toolkit currently
    target_column: str
    folder: Path


class CarInputs2(BaseConfig):
    x: Optional[Path] = None
    y: Optional[Path] = None
    folder_path: Optional[Path] = None
    index_columns: Optional[List[str]] = None
    drop_columns: Optional[List[str]] = None
    wide_format: Optional[str] = None
    variable_name: Optional[str] = None
    value_name: Optional[str] = None
    model_type: Any  # abc.ABCMeta not supported by caf.toolkit currently
    target_column: Optional[str] = None
    folder: Optional[Path] = None
    output_folder: Optional[Path] = None
    outlier_threshold: Optional[str] = None
    cv_method: Optional[str] = None
    splits: Optional[str] = None
    repeats: Optional[str] = None
    hp_optimisation: Optional[str] = None


class LvuLog(BaseConfig):
    inputs: LvuInputs

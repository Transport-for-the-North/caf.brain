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
from typing import Any, Optional


# # # CONSTANTS # # #
# # # CLASSES # # #
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
    index1: Optional[str] = None
    index2: Optional[str] = None
    wide_format: Optional[str] = None
    variable_name: Optional[str] = None
    value_name: Optional[str] = None
    method: Any  # abc.ABCMeta not supported by caf.toolkit currently
    target_column: str
    folder: Optional[Path] = None


class LvuLog(BaseConfig):
    inputs: LvuInputs

# # # FUNCTIONS # # #

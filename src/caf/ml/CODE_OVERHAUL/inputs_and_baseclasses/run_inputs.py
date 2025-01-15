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
from caf.toolkit import BaseConfig


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

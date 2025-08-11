# -*- coding: utf-8 -*-
"""
Created on: 1/24/2025
Original author: Adil Zaheer
"""
# Built-Ins
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from pathlib import Path
from typing import Any, List, Optional, Union

# Third Party
from caf.toolkit import BaseConfig


class TensorFlowModelInputs(BaseConfig):
    training_data: Optional[Path] = None
    prediction_data: Optional[Path] = None
    validation_data: Optional[Path] = None
    output_folder: Optional[Path] = None
    target_column: Optional[str] = None
    index_columns: Optional[List[str]] = None
    drop_columns: Optional[List[str]] = None
    categorical_target: Optional[str] = None
    column_name_to_drop_rows: Optional[List[str]] = None
    value_in_row: Optional[List[Union[str, int, float]]] = None
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    simple_model: Optional[str] = None

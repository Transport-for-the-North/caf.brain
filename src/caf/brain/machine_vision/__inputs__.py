# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 3/3/2025
Original author: Adil Zaheer
"""
from pathlib import Path
from typing import Optional
from caf.toolkit import BaseConfig


class MachineVisionInputs(BaseConfig):
    training_data: Optional[Path] = None
    output_path: Optional[Path] = None
    noham_path: Optional[Path] = None
    geo_path: Optional[Path] = None

# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 3/4/2025
Original author: Adil Zaheer
"""
from caf.brain.machine_vision.noham_processing.noham_processing_functions import (
    noham_data,
)


def main_process_noham(output_path, noham_path, geo_path):

    noham_dat = noham_data(geo_path=geo_path, noham_path=noham_path, output=output_path)

    return noham_dat

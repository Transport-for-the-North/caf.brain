# -*- coding: utf-8 -*-
"""
Created on: 1/15/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.CODE_OVERHAUL.MODELS.NorCom.__main__ import main
from src.caf.ml.CODE_OVERHAUL.inputs_and_baseclasses.run_inputs import run_file_inputs
import yaml


with open('config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

params = run_file_inputs(**config_data)


if __name__ == '__main__':
    main(params)

# -*- coding: utf-8 -*-
"""
Created on: 12/16/2024
Original author: Adil Zaheer
"""

# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.CODE_OVERHAUL.process_data_functions.process_data_main import main_input_data
from src.caf.ml.CODE_OVERHAUL.inputs_and_baseclasses.run_inputs import run_file_inputs
from pathlib import Path


if __name__ == '__main__':
    params = run_file_inputs(
                             file_path=Path(r"E:\2025 work streams\redo_cafml\cb_tfn_v15.csv"),
                             folder_path=None,
                             output_path=Path(r'E:\2025 work streams\redo_cafml'),

                             # # # PROCESSING INPUT DATA # # #
                             target_column='numcarvan',
                             custom_index=['householdid', 'surveyyear', 'hholdua_b01id'],
                             column_name_to_drop_rows=['tfn_at'],
                             value_in_row=['20'],

                             # # # TRANSFORMING DATA # # #
                             full_transformations=False,
                             categorical_features=['tfn_at', 'hh_child', 'ns', 'hholdnumadults'],
                             numerical_features=['trav_dist'],
                             weight_column='w2',
                             binary_prediction='multiclass',
                             time_series_split=None,

                             # # # MODELLING # # #
                             model_choice=None

    )

    main_input_data(params)

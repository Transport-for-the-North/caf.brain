# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 2/18/2025
Original author: Adil Zaheer
"""
import time
import os
import logging
from caf.brain.machine_vision.junction_detection_functions.noham_processing.noham_processing_main import main_process_noham
from caf.brain.machine_vision.satellite_image_processing.input_processing.input_processing_functions import process_file
from caf.brain.machine_vision.satellite_image_processing.input_processing.input_processing_main import main_input_processing
LOG = logging.getLogger(__name__)


def main(params):
    start_time = time.time()

    output_path = os.path.join(params.output_path, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if params.generate_training_images:
        if params.noham_path and params.geo_path is not None:
            LOG.info("It is recommended to process any Noham data outside of the model \
                      the model will attempt to process the data but this is based on the \
                      NoHam 2023 base database file and the NoHam base network node shapefile.")

            # todo just put catch inside main_process_noham no need for if else here
            training_data_info = main_process_noham(output_path=output_path,
                                                    noham_path=params.noham_path,
                                                    geo_path=params.geo_path)

            main_input_processing(geo_df=None,
                                  df=training_data_info,
                                  image_folder=params.image_folder_path,
                                  final_html_info=params.final_html_info,
                                  x_coordinate='coordinates_easting',
                                  y_coordinate='coordinates_northing',
                                  output_path=output_path)

        else:
            LOG.info("Reading in training data information. This must be built from \
                      the NoHam base network outside of this model or using the noham_data \
                      function which requires the NoHam base network database and shapefiles.")

            training_data_info, file_type = process_file(params.training_data_information)

            if file_type == '.csv':
                main_input_processing(geo_df=None,
                                      df=training_data_info,
                                      image_folder=params.image_folder_path,
                                      final_html_info=params.final_html_info,
                                      x_coordinate=params.x_column,
                                      y_coordinate=params.y_column,
                                      output_path=output_path)
            else:
                main_input_processing(geo_df=training_data_info,
                                      df=None,
                                      image_folder=params.image_folder_path,
                                      final_html_info=params.final_html_info,
                                      x_coordinate=params.x_column,
                                      y_coordinate=params.y_column,
                                      output_path=output_path)

    end_time = time.time()
    LOG.info(f"Total run time: {end_time - start_time:.2f} seconds")

    return

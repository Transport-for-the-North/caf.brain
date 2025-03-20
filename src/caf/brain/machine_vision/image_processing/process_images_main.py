# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 3/10/2025
Original author: Adil Zaheer
"""
import pandas as pd
import os
from caf.brain.machine_vision.image_processing.image_processing_functions import euclidean_distance, directory_iterator, image_crop


def main_process_images(html_data,
                        noham_data,
                        output,
                        training_data_path):
    # calc ecu distance, finalise list of satellite images we actually need DONE
    # read in one by one, auto zoom to area and export
    # manually label with label maker

    final_input_data, satellites = euclidean_distance(df_a=html_data,
                                                      df_b=noham_data)

    final_input_data = pd.merge(final_input_data, html_data, on='box_boundary', how='inner')
    final_input_data.to_csv(os.path.join(output, 'final_input_data.csv'), index=False)

    path_list = directory_iterator(dir_path=training_data_path,
                                   training_data=final_input_data,
                                   image_reference_column='box_boundary')

    image_crop(path_list=path_list,
               image_supporting_data=final_input_data,
               output=output)

    return

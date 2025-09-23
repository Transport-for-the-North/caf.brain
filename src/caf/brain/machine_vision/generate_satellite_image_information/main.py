"""
Created on: 9/17/2025
Original author: Adil Zaheer
"""
import os
import pandas as pd
import logging
from pathlib import Path

from caf.brain.machine_vision.generate_satellite_image_information.functions import image_path_name_finder, satellite_xml_processor

LOG = logging.getLogger(__name__)


def image_info_generation(output_path: Path,
                          image_folder_path: Path):
    df_filename = os.path.join(output_path, 'satellite_image_metadata.csv')
    if os.path.exists(df_filename):
        LOG.info('Satellite image metadata already exists and is being read in')
        image_metadata = pd.read_csv(df_filename)
    else:
        image_xml_df = satellite_xml_processor(folder_path=image_folder_path,
                                               output_path=output_path)

        image_path_df = image_path_name_finder(folder_path=image_folder_path)

        image_metadata = pd.merge(image_xml_df, image_path_df, on='box_boundary', how='left')
        image_metadata.to_csv(df_filename, index=False)
        LOG.info('Satellite image metadata saved to: %', df_filename)

    return image_metadata

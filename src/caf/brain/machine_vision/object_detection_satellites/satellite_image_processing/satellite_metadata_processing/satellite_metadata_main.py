"""
Created on: 7/22/2025
Original author: Adil Zaheer
"""

import os
import logging
from pathlib import Path
import pandas as pd
from caf.brain.machine_vision.object_detection_satellites.satellite_image_processing.satellite_metadata_processing.satellite_metadata_functions import (
    create_satellite_image_metadata,
)

LOG = logging.getLogger(__name__)
from caf.brain.machine_vision.object_detection_satellites.satellite_image_processing.satellite_metadata_processing.xml_processing_functions import (
    main_process_xml,
)


def main_satellite_metadata(
    folder_path: Path, output_path: Path, satellite_image_metadata: Path = None
) -> pd.DataFrame:
    LOG.info("Satellite image metadata creation beginning")
    df_filename = os.path.join(output_path, "satellite_image_metadata.csv")

    if satellite_image_metadata:
        LOG.info("Satellite image metadata path provided and is being read in")
        df = pd.read_csv(os.path.join(satellite_image_metadata))
        return df

    if os.path.exists(df_filename):
        LOG.info("Satellite image metadata already exists and is being read in")
        df = pd.read_csv(df_filename)
        return df

    image_xml_df = main_process_xml(folder_path=folder_path, output_path=output_path)

    image_path_df = create_satellite_image_metadata(folder_path=folder_path)

    metadata = pd.merge(image_xml_df, image_path_df, on="box_boundary", how="left")
    metadata.to_csv(df_filename, index=False)
    return metadata

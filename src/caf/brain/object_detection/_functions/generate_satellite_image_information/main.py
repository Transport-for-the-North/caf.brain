"""
Created on: 9/17/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import os
from pathlib import Path

# Third Party
import pandas as pd

# Local Imports
from caf.brain.object_detection._functions.generate_satellite_image_information.functions import (
    _image_path_name_finder,
    _satellite_xml_processor,
)

LOG = logging.getLogger(__name__)


def image_info_generation(output_path: Path, image_folder_path: Path) -> pd.DataFrame:
    """
    Generates satellite image metadata for each British National Grid tile.

    Parameters
    ----------
    output_path:
        Path to output folder.
    image_folder_path:
        Path to folder that contains British National Grid tile jpegs and their
        accompanying XML files.

    Returns
    -------
    satellite_image_metadata:
        Satellite image metadata containing tile names, midpoints and path
        locations.

    """
    output_path = Path(output_path)
    image_folder_path = Path(image_folder_path)

    df_filename = output_path / "satellite_image_metadata.csv"

    if os.path.exists(df_filename):
        LOG.info("Satellite image metadata already exists and is being read in")
        satellite_image_metadata = pd.read_csv(df_filename)
    else:
        image_xml_df = _satellite_xml_processor(
            image_folder_path=image_folder_path, output_path=output_path
        )

        image_path_df = _image_path_name_finder(folder_path=image_folder_path)

        satellite_image_metadata = pd.merge(
            image_xml_df, image_path_df, on="box_boundary", how="left"
        )
        satellite_image_metadata.to_csv(df_filename, index=False)
        LOG.info("Satellite image metadata saved here: %s", df_filename)

    return satellite_image_metadata

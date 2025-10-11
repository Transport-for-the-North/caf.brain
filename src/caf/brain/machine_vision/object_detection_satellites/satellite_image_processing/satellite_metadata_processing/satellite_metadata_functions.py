"""
Created on: 7/22/2025
Original author: Adil Zaheer
"""

from pathlib import Path
import pandas as pd
import os
import glob


def create_satellite_image_metadata(folder_path: Path) -> pd.DataFrame:
    """
    Helper function for finding satellite image names and path locations.

    Parameters
    ----------
    folder_path: Path to folder that contains the satelite images (JPG).

    Returns
    -------
    df: Dataframe of image paths and their box boundary (BNG name).
    """
    meta_dict = {}

    paths = glob.glob(os.path.join(folder_path) + "/**/*.jpg", recursive=True)
    for path in paths:
        directory, file_name = os.path.split(path)
        name = os.path.splitext(file_name)[0]
        meta_dict[name] = path

    df = pd.DataFrame(list(meta_dict.items()), columns=["box_boundary", "path"])

    return df

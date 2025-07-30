"""
Created on: 3/4/2025
Original author: Adil Zaheer
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import rasterio
from rasterio import RasterioIOError
import logging

LOG = logging.getLogger(__name__)


def euclidean_distance(df_a, df_b):
    """
    df_a: html info about satellites
    df_b: their coordinates they want to have satellites images for
    """
    coords_a = df_a[['tile_easting', 'tile_northing']].values
    coords_b = df_b[['coordinates_easting', 'coordinates_northing']].values

    # Euclidean distance between each pair of points
    distances = cdist(coords_b, coords_a, metric='euclidean')

    closest_indices = np.argmin(distances, axis=1)
    closest_boundaries = df_a.iloc[closest_indices]['box_boundary'].values

    result_df = df_b.copy()
    result_df['box_boundary'] = closest_boundaries

    satellites = result_df['box_boundary'].unique()
    satellites = pd.DataFrame(satellites, columns=['BNG_tile_names'])

    return result_df, satellites


def check_raster_file(path):
    try:
        with rasterio.open(path) as img:
            return True
    except RasterioIOError as e:
        print(f"Error opening file {path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error with {path}: {e}")
        return False

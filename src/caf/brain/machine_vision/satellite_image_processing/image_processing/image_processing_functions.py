# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 3/4/2025
Original author: Adil Zaheer
"""
import glob
import pandas as pd
from tqdm import tqdm
import os
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


def directory_iterator(dir_path: Path,
                       training_data,
                       image_reference_column: str,
                       output: Path,
                       folder_if_loop: Path) -> list[Path]:

    path_file_name = "satellite_image_paths.csv"
    if os.path.exists(os.path.join(folder_if_loop, path_file_name)):
        df = pd.read_csv(os.path.join(folder_if_loop, path_file_name))
        path_list = df.iloc[:, 0].tolist()
        return path_list

    else:
        counter = 0
        paths = glob.glob(os.path.join(dir_path) + '/**/*.jpg', recursive=True)

        try:
            training = pd.read_csv(training_data)
        except TypeError:
            training = training_data

        training[image_reference_column] = training[image_reference_column].astype(str)
        reference_set = set(training[image_reference_column].values)

        path_list = []
        LOG.info(f"Looking for {len(reference_set)} unique reference values in {len(paths)} image files")

        with tqdm(total=None) as pbar:
            for path in paths:
                directory, file_name = os.path.split(path)
                name = os.path.splitext(file_name)[0]

                if name in reference_set:
                    path_list.append(path)
                pbar.update(1)
                counter += 1
                if counter % 100 == 0:
                    LOG.info(f"Processed {counter}/{len(paths)} paths, found {len(path_list)} matches so far")

        path_list_df = pd.DataFrame(path_list)
        path_list_df.to_csv(os.path.join(folder_if_loop, path_file_name), index=False)

    return path_list


def create_satellite_image_metadata(dir_path: Path,
                                    output: Path):
    path_file_name = "satellite_image_metadata.csv"
    meta_dict = {}

    if os.path.exists(os.path.join(output, path_file_name)):
        df = pd.read_csv(os.path.join(output, path_file_name))
        return df

    else:
        paths = glob.glob(os.path.join(dir_path) + '/**/*.jpg', recursive=True)
        for path in paths:
            directory, file_name = os.path.split(path)
            name = os.path.splitext(file_name)[0]
            meta_dict[name] = path

        df = pd.DataFrame(list(meta_dict.items()), columns=['box_boundary', 'path'])
        df.to_csv(os.path.join(output, path_file_name), index=False)

    return df


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


# def find_surrounding_names(file_name) -> dict:
#     file_name_list = list(file_name)
#     prefix = file_name_list[:2]
#     prefix = "".join(prefix)
#
#     nums = file_name_list[2:]
#     concat_nums = ''.join(nums)
#     nums_int = int(concat_nums)
#
#     east = nums_int + 100
#     west = nums_int - 100
#     north = nums_int + 1
#     south = nums_int - 1
#     northeast = nums_int + 101
#     northwest = nums_int - 99
#     southeast = nums_int + 99
#     southwest = nums_int - 101
#     # TODO need to make this more robust to handle crossing boundaries etc. i think this is causing a problem as
#     # todo some box_boundary names are only 3 numbers
#     # todo also when it gets to the final new mosaic image it isn't writing out properly? empty image?
#     image_layout_dict = {'centre_image': file_name,
#                          'north': north,
#                          'south': south,
#                          'east': east,
#                          'west': west,
#                          'northeast': northeast,
#                          'southeast': southeast,
#                          'southwest': southwest,
#                          'northwest': northwest}
#
#     for key in image_layout_dict:
#         if key != 'centre_image':
#             image_layout_dict[key] = prefix + str(image_layout_dict[key])
#
#     return image_layout_dict

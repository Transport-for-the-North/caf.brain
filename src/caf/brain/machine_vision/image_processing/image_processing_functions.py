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
import cv2
import joblib
import numpy as np
from scipy.spatial.distance import cdist
import rasterio
from rasterio.windows import Window
from PIL import Image
from pathlib import Path
import logging
LOG = logging.getLogger(__name__)


# def read_and_store_images(folder_path, output_path):
#     LOG.info("Reading in satellite images")
#     data_filename = os.path.join(output_path, 'img_dict.pkl')
#
#     if os.path.exists(data_filename):
#         LOG.info('Satellite image dictionary already exists and is being read in')
#         images_dict = joblib.load(data_filename)
#     else:
#         images_dict = {}
#
#         counter = 0
#         paths = glob.glob(os.path.join(folder_path) + '/**/*.jpg', recursive=True)
#         with tqdm(total=None) as pbar:
#             for path in paths:
#                 img = cv2.imread(path)
#                 directory, file_name = os.path.split(path)
#                 name = os.path.splitext(file_name)[0]
#
#                 images_dict[name] = img
#                 pbar.update(1)
#                 counter += 1
#                 if counter % 1000 == 0:
#                     LOG.info(f"Processed item {counter}: {path}")
#         joblib.dump(images_dict, data_filename)
#
#     LOG.info("Satellite images successfully read in")
#     return images_dict


def euclidean_distance(df_a, df_b):
    coords_a = df_a[['tile_easting', 'tile_northing']].values
    coords_b = df_b[['junction_easting', 'junction_northing']].values

    # Euclidean distance between each pair of points
    distances = cdist(coords_b, coords_a, metric='euclidean')

    closest_indices = np.argmin(distances, axis=1)
    closest_boundaries = df_a.iloc[closest_indices]['box_boundary'].values

    result_df = df_b.copy()
    result_df['box_boundary'] = closest_boundaries

    satellites = result_df['box_boundary'].unique()
    satellites = pd.DataFrame(satellites, columns=['BNG_tile_names'])

    return result_df, satellites


def image_crop(path_list,
               image_supporting_data,
               output):
    """
    - must be geo-referenced data

    """
    output_dir = os.path.join(output, 'images_for_labelling')
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    invalid_count = 0
    invalid_images = []

    with tqdm(total=None) as pbar:
        for path in path_list:
            # get file name (image) for saving
            base_name = os.path.basename(path)
            file_name = os.path.splitext(base_name)[0]

            row_data = image_supporting_data[image_supporting_data['box_boundary'] == file_name]

            if row_data.empty:
                LOG.warning(f"No matching data found for {file_name}")
                pbar.update(1)
                continue

            junction_easting = row_data['junction_easting'].values[0]
            junction_northing = row_data['junction_northing'].values[0]

            with rasterio.open(path) as img:
                transform = img.transform
                # bng to pixels
                col, row = ~transform * (junction_easting, junction_northing)  # col and row are pixel coords of target

                # crop window size in pixels
                pixel_resolution = abs(transform[0])  # (find width of pixel in meters)
                crop_size_pixels = int(200 / pixel_resolution)

                # create crop area
                half_size = crop_size_pixels // 2  # half size used so half junc is in the middle of the crop (half one side, half the other)

                # Calculate the initial crop window coordinates
                # col_start and row_start are top left
                # col_end and row_end are bottom right
                col_start = int(col - half_size)  # left edge of ideal window
                row_start = int(row - half_size)  # top edge of ideal window
                col_end = int(col + half_size)  # right edge of ideal window
                row_end = int(row + half_size)  # bottom edge of ideal window

                # Accounting for image boundaries (some junctions are not in the middle of an image)
                # first two return larger (if col/row_start are negative, then set to 0)
                # second two return smaller (if col/row_end exceed image bounds, set to actual image bounds)
                col_start = max(0, col_start)  # Ensure left edge isn't outside image
                row_start = max(0, row_start)  # Ensure top edge isn't outside image
                col_end = min(img.width, col_end)  # Ensure right edge doesn't exceed image width
                row_end = min(img.height,
                              row_end)  # Ensure bottom edge doesn't exceed image height

                # Calculate actual width and height after boundary adjustment
                actual_width = col_end - col_start
                actual_height = row_end - row_start

                if actual_width <= 0 or actual_height <= 0:
                    LOG.warning(f"Invalid dimensions calculated for {path}")
                    LOG.warning(f"Target point: col={col}, row={row}")
                    LOG.warning(f"Adjusted window: col_start={col_start}, col_end={col_end}, row_start={row_start}, row_end={row_end}")
                    LOG.warning(f"Image dimensions: width={img.width}, height={img.height}")

                    directory, file_name = os.path.split(path)
                    name = os.path.splitext(file_name)[0]

                    invalid_images.append(name)

                    invalid_count += 1
                    LOG.info(f'Total invalid images: {invalid_count}')
                    continue

                # Create the window with adjusted coordinates
                window = Window(col_start,  # left edge of box
                                row_start,  # top of box
                                actual_width,  # width
                                actual_height)  # height

                # creates 3d array (bands (2d array of pixels, 3 bands for rgb), height, width)
                cropped_img = img.read(window=window)

                # create pil image from array with correct format and value order
                # rearrange array into height width bands format for PIL image
                final_image = Image.fromarray(np.moveaxis(cropped_img, 0, -1).astype(np.uint8))

                output_filename = os.path.join(output_dir, f"{file_name}_cropped.jpg")
                final_image.save(output_filename, "JPEG", quality=95)

                pbar.update(1)
                count += 1
                if count % 100 == 0:
                    LOG.info(f"Processed item {count}: {path}")

    df = pd.DataFrame(invalid_images, columns=['invalid_junctions'])
    df.to_csv(os.path.join(output, 'failed_image_crops.csv'), index=False)

    return


def directory_iterator(dir_path: Path,
                       training_data,
                       image_reference_column: str) -> list[Path]:

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

    return path_list




# todo make all this code more efficient overall
# todo, extract coordinates from geometry column
# todo match the coordinates with corresponding british national grid tile
# todo may combine this data with the html stuff? one big data set with corresponding image under the bng tile grid as the dict key?
# todo then i know which junctions (noham base network nodes) are where then can zoom the images
# todo then can start labelling them


# inloop create df that has all the





# junctions
# PM = motorway merge
# R = roundabout
# P = give way
# EP = give way on approach to roundabout
# S = signalised junc
# EX = DROP THIS
# ES = traffic signal on approach to roundabout
# M = mini roundabout
# p = same as P
# s = same as S


# nodes less than 10k are zone connectors (not relevant)


# JUNCTION: ignore None / Z
# ICD (diameter of roundabout), will only have a value if its a roundabout
# drop ucycle, gap, gapm
# MIDLANE = no lanes between one junction and next. so on the way to the junction
# STPLANE = no lanes at stop line of the junction. 1f means two lanes at the junction but one lane is small
# MAJOR = if Y then the road that the major is on is the main road. if signalised, major minor doesnt apply. major minor only applies to give way or give way roundabout

# turns:
# turns work from left to right. so turn one will always be in the left lane
# first number is lane index you can use to turn. second number is how many lanes can turn
#l1-1 means left most lane and only one lane to turn left
# a for ahead. a1-1 means one lane going ahead and you can only use one lane to go ahead

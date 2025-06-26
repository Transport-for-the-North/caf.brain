# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 4/27/2025
Original author: Adil Zaheer
"""
import pandas as pd
import os
import numpy as np
import rasterio
from PIL import Image
from rasterio.merge import merge
import logging
from rasterio.errors import MergeError
from caf.brain.machine_vision.satellite_image_processing.image_processing.image_processing_functions import \
    check_raster_file
LOG = logging.getLogger(__name__)


def find_surrounding_images(col_start,
                            row_start,
                            col_end,
                            row_end,
                            image_width,
                            image_height):

    missing_left = 0 if col_start >= 0 else abs(col_start)
    missing_top = 0 if row_start >= 0 else abs(row_start)
    missing_right = 0 if col_end <= image_width else col_end - image_width
    missing_bottom = 0 if row_end <= image_height else row_end - image_height

    needed_tiles = []

    if missing_top > 0:
        needed_tiles.append("north")
    if missing_right > 0:
        needed_tiles.append("east")
    if missing_bottom > 0:
        needed_tiles.append("south")
    if missing_left > 0:
        needed_tiles.append("west")

    if missing_top > 0 and missing_left > 0:
        needed_tiles.append("northwest")
    if missing_top > 0 and missing_right > 0:
        needed_tiles.append("northeast")
    if missing_bottom > 0 and missing_left > 0:
        needed_tiles.append("southwest")
    if missing_bottom > 0 and missing_right > 0:
        needed_tiles.append("southeast")

    return needed_tiles


def find_surrounding_names(file_name):

    # west to east, south to north
    grid_letters = [
        ['SV', 'SW', 'SX', 'SY', 'SZ', 'TV', 'TW'],
        ['SQ', 'SR', 'SS', 'ST', 'SU', 'TQ', 'TR'],
        ['SL', 'SM', 'SN', 'SO', 'SP', 'TL', 'TM'],
        ['SF', 'SG', 'SH', 'SJ', 'SK', 'TF', 'TG'],
        ['SA', 'SB', 'SC', 'SD', 'SE', 'TA', 'TB'],
        ['NV', 'NW', 'NX', 'NY', 'NZ', 'OV', 'OW'],
        ['NQ', 'NR', 'NS', 'NT', 'NU', 'OQ', 'OR'],
        ['NL', 'NM', 'NN', 'NO', 'NP', 'OL', 'OM'],
        ['NF', 'NG', 'NH', 'NJ', 'NK', 'OF', 'OG'],
        ['NA', 'NB', 'NC', 'ND', 'NE', 'OA', 'OB'],
        ['HV', 'HW', 'HX', 'HY', 'HZ', 'JV', 'JW'],
        ['HQ', 'HR', 'HS', 'HT', 'HU', 'JQ', 'JR'],
        ['HL', 'HM', 'HN', 'HO', 'HP', 'JL', 'JM']
    ]

    prefix = file_name[:2]
    nums_str = file_name[2:]

    if len(nums_str) != 4:
        LOG.error(f"Expected 4-digit number after prefix, got {nums_str} \
                    for {file_name}.")
        raise ValueError(f"Expected 4-digit number after prefix, got {nums_str}")

    easting_major = int(nums_str[0:2])
    northing_major = int(nums_str[2:4])

    # locate where file_name prefix is in our bng grid (helps when during border crossing)
    prefix_row = None
    prefix_col = None
    for row_idx, row in enumerate(grid_letters):
        if prefix in row:
            prefix_row = row_idx
            prefix_col = row.index(prefix)
            break

    if prefix_row is None or prefix_col is None:
        raise ValueError(f"Prefix {prefix} not found in the grid letters map")

    def get_adjusted_reference(row_change, col_change, east_change, north_change):
        new_row = prefix_row - row_change  # Subtract because rows increase southward in our grid
        new_col = prefix_col + col_change

        new_easting = easting_major + east_change
        new_northing = northing_major + north_change

        if new_easting >= 100:
            new_easting -= 100
            new_col += 1
        elif new_easting < 0:
            new_easting += 100
            new_col -= 1

        if new_northing >= 100:
            new_northing -= 100
            new_row -= 1  # Going north means decreasing row in our grid
        elif new_northing < 0:
            new_northing += 100
            new_row += 1  # Going south means increasing row in our grid

        # Ensure we're still within the valid grid
        if 0 <= new_row < len(grid_letters) and 0 <= new_col < len(grid_letters[0]):
            new_prefix = grid_letters[new_row][new_col]
            return f"{new_prefix}{new_easting:02d}{new_northing:02d}"
        else:
            return None  # gone off the edge of our defined grid

    image_layout_dict = {
        'centre_image': file_name,
        'north': get_adjusted_reference(0, 0, 0, 1),
        'south': get_adjusted_reference(0, 0, 0, -1),
        'east': get_adjusted_reference(0, 0, 1, 0),
        'west': get_adjusted_reference(0, 0, -1, 0),
        'northeast': get_adjusted_reference(0, 0, 1, 1),
        'northwest': get_adjusted_reference(0, 0, -1, 1),
        'southeast': get_adjusted_reference(0, 0, 1, -1),
        'southwest': get_adjusted_reference(0, 0, -1, -1)
    }

    return image_layout_dict


def surrounding_img_path_finder(image_layout_dict,
                                list_of_needed_tiles,
                                image_folder,
                                satellite_metadata: pd.DataFrame):

    images_to_concat = []
    final_images_to_concat = []

    for key, val in image_layout_dict.items():
        for item in list_of_needed_tiles:
            if item == key:
                images_to_concat.append(val)

    for item in images_to_concat:
        if item in satellite_metadata['box_boundary'].values:
            row_data = satellite_metadata[satellite_metadata['box_boundary'] == item]
            path = row_data['path'].values[0]
            final_images_to_concat.append(path)

    if len(images_to_concat) != len(final_images_to_concat):
        LOG.warning("Not all tiles required for image expansion are available")
        final_images_to_concat = []
        return final_images_to_concat

    return final_images_to_concat


def estimate_memory_of_mosaic(image_paths):
    bounds_list = []
    paths_not_fine = []

    for path in image_paths:
        if not check_raster_file(path):
            paths_not_fine.append(path)

    if paths_not_fine:
        return None
    else:
        for path in image_paths:
            with rasterio.open(path) as ds:
                bounds_list.append(ds.bounds)
                if len(bounds_list) == 1:
                    res_x, res_y = ds.res
                    num_bands = ds.count

        # total bounds
        min_left = min(bound.left for bound in bounds_list)
        min_bottom = min(bound.bottom for bound in bounds_list)
        max_right = max(bound.right for bound in bounds_list)
        max_top = max(bound.top for bound in bounds_list)

        # dimensions in pixels
        est_width = int((max_right - min_left) / res_x)
        est_height = int((max_top - min_bottom) / res_y)

        # memory requirement in bytes, then gigabytes
        bytes_per_element = np.dtype(np.uint8).itemsize
        est_memory_bytes = est_width * est_height * num_bands * bytes_per_element
        est_memory_gb = est_memory_bytes / (1024 ** 3)

        return est_memory_gb


def create_new_image(image_paths_to_concat,
                     centre_image_path,
                     junction_easting,
                     junction_northing,
                     output_dir,
                     file_name):
    # todo more eloquent memory storage solution

    est_mem = estimate_memory_of_mosaic(image_paths=image_paths_to_concat)
    if est_mem is None:
        LOG.error("Issue with one or all paths required for mosaic. Image expansion not possible.")
        return
    if est_mem > 30:
        LOG.error(f"Could not create mosaic for {centre_image_path} due to memory allocation")
        return

    datasets = []
    for path in image_paths_to_concat:
        ds = rasterio.open(path)
        datasets.append(ds)

    centre_ds = rasterio.open(centre_image_path)
    datasets.append(centre_ds)

    # Merge datasets. returns np array (mosaic) and the transform
    try:
        mosaic, mosaic_transform = merge(datasets)
    except MergeError as e:
        print(f"Error merging mosaic: {e}")
        return

    if mosaic is None or mosaic.size == 0:
        LOG.error(f"Mosaic creation failed for {centre_image_path} - empty or null mosaic returned")
        return

    # Convert junction coordinates (in map space) to pixel
    col, row = ~mosaic_transform * (junction_easting, junction_northing)

    crop_size_pixels = 640
    half_size = crop_size_pixels // 2

    col_start = int(col - half_size)
    row_start = int(row - half_size)
    col_end = int(col + half_size)
    row_end = int(row + half_size)

    if (col_start < 0 or row_start < 0 or
        col_end > mosaic.shape[2] or row_end > mosaic.shape[1]):
        LOG.warning(f"{centre_image_path} crop extends beyond mosaic boundaries:")
        LOG.warning(f" Mosaic shape: {mosaic.shape}")
        LOG.warning(f" Crop window: ({row_start}:{row_end}, {col_start}:{col_end})")
        return

    # mosaic has shape (bands, height, width)
    cropped_img = mosaic[:, row_start:row_end, col_start:col_end]

    expected_shape = (mosaic.shape[0], crop_size_pixels, crop_size_pixels)
    if cropped_img.shape != expected_shape:
        LOG.warning(f" {centre_image_path }cropped image has unexpected dimensions:")
        LOG.warning(f" Expected: {expected_shape}")
        LOG.warning(f" Actual: {cropped_img.shape}")
        return

    # Convert the image: move bands to the last axis and cast to uint8 for JPEG
    final_image = Image.fromarray(np.moveaxis(cropped_img, 0, -1).astype(np.uint8))
    output_filename = os.path.join(output_dir, f"{file_name}_cropped_extended.jpg")
    final_image.save(output_filename, "JPEG", quality=95)

    for ds in datasets:
        ds.close()

    return

"""
Created on: 10/6/2025
Original author: Adil Zaheer
"""

from pathlib import Path
import os
import logging
import pandas as pd
import numpy as np
import rasterio
from PIL import Image
from rasterio.merge import merge
from rasterio.errors import MergeError, RasterioIOError, RasterioError

LOG = logging.getLogger(__name__)


def _find_surrounding_images(
    col_start: int,
    row_start: int,
    col_end: int,
    row_end: int,
    image_width: int,
    image_height: int,
) -> list:
    """
    Helper function for finding which British National Grid tiles are required
    to create a full cropped image.

    Parameters
    ----------
    col_start: Left edge of ideal crop window.
    row_start: Top edge of ideal crop window.
    col_end: Right edge of ideal crop window.
    row_end: Bottom edge of ideal crop window.
    image_width: Full input image width.
    image_height: Full input image height.

    Returns
    -------
    needed_tiles: List of what surrounding tiles are required for the ideal
                  crop.
    """

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


def _find_surrounding_names(file_name: str) -> dict:
    """
    Provides the names of all the surrounding British National Grid tiles.

    Parameters
    ----------
    file_name: Your British National Grid tile that contains the point of
               interest. This name should be taken from the image jpeg name.

    Returns
    -------
    image_layout_dict: All surrounding British National Grid tiles in each
                       direction.
    """

    # west to east, south to north
    grid_letters = [
        ["SV", "SW", "SX", "SY", "SZ", "TV", "TW"],
        ["SQ", "SR", "SS", "ST", "SU", "TQ", "TR"],
        ["SL", "SM", "SN", "SO", "SP", "TL", "TM"],
        ["SF", "SG", "SH", "SJ", "SK", "TF", "TG"],
        ["SA", "SB", "SC", "SD", "SE", "TA", "TB"],
        ["NV", "NW", "NX", "NY", "NZ", "OV", "OW"],
        ["NQ", "NR", "NS", "NT", "NU", "OQ", "OR"],
        ["NL", "NM", "NN", "NO", "NP", "OL", "OM"],
        ["NF", "NG", "NH", "NJ", "NK", "OF", "OG"],
        ["NA", "NB", "NC", "ND", "NE", "OA", "OB"],
        ["HV", "HW", "HX", "HY", "HZ", "JV", "JW"],
        ["HQ", "HR", "HS", "HT", "HU", "JQ", "JR"],
        ["HL", "HM", "HN", "HO", "HP", "JL", "JM"],
    ]

    prefix = file_name[:2]
    nums_str = file_name[2:]

    if len(nums_str) != 4:
        LOG.error("Expected 4-digit number after prefix, got %s for %s.", nums_str, file_name)
        raise ValueError(f"Expected 4-digit number after prefix, got {nums_str}")

    easting_major = int(nums_str[0:2])
    northing_major = int(nums_str[2:4])

    # locate where file_name prefix is in our bng grid (helps during border crossing)
    prefix_row = None
    prefix_col = None
    for row_idx, row in enumerate(grid_letters):
        if prefix in row:
            prefix_row = row_idx
            prefix_col = row.index(prefix)
            break

    if prefix_row is None or prefix_col is None:
        raise ValueError(f"Prefix {prefix} not found in the grid letters map")

    def _get_adjusted_reference(
        row_change: int, col_change: int, east_change: int, north_change: int
    ):
        """
        Calculates the adjusted British National Grid tile name based on
        directional shifts. This is calculated  by applying changes to the row,
        column, easting, and northing values of the original tile.

        Parameters
        ----------
        row_change: Change in grid row index (positive = southward, negative = northward).
        col_change : Change in grid column index (positive = eastward, negative = westward).
        east_change : Change in easting within the tile (0–99).
        north_change : Change in northing within the tile (0–99).

        Returns
        -------
        The adjusted tile name in British National Grid format or None if the
        adjustment moves outside the defined grid.
        """
        # Subtract because rows increase southward in our grid
        new_row = prefix_row - row_change
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

        return None  # gone off the edge of our defined grid

    image_layout_dict = {
        "centre_image": file_name,
        "north": _get_adjusted_reference(0, 0, 0, 1),
        "south": _get_adjusted_reference(0, 0, 0, -1),
        "east": _get_adjusted_reference(0, 0, 1, 0),
        "west": _get_adjusted_reference(0, 0, -1, 0),
        "northeast": _get_adjusted_reference(0, 0, 1, 1),
        "northwest": _get_adjusted_reference(0, 0, -1, 1),
        "southeast": _get_adjusted_reference(0, 0, 1, -1),
        "southwest": _get_adjusted_reference(0, 0, -1, -1),
    }

    return image_layout_dict


def _surrounding_img_path_finder(
    image_layout_dict: dict, list_of_needed_tiles: list, satellite_metadata: pd.DataFrame
) -> list:
    """
    Locates paths of images required for the final image crop based on what
    surrounding image tiles are available.

    Parameters
    ----------
    image_layout_dict: All surrounding British National Grid tiles in each
                       direction, generated from _find_surrounding_names.
    list_of_needed_tiles: List of what surrounding tiles are required for the
                          ideal crop, generated from _find_surrounding_images.
    satellite_metadata: Satellite image metadata containing tile names,
                        midpoints and path locations, generated from
                        image_info_generation.

    Returns
    -------
    final_images_to_concat: List of final images to combine to get the ideal
                            crop window.
    """

    images_to_concat = []
    final_images_to_concat = []

    for key, val in image_layout_dict.items():
        for item in list_of_needed_tiles:
            if item == key:
                images_to_concat.append(val)

    for item in images_to_concat:
        if item in satellite_metadata["box_boundary"].values:
            row_data = satellite_metadata[satellite_metadata["box_boundary"] == item]
            path = row_data["path"].values[0]
            final_images_to_concat.append(path)

    if len(images_to_concat) != len(final_images_to_concat):
        LOG.warning("Not all tiles required for image expansion are available")
        final_images_to_concat = []
        return final_images_to_concat

    return final_images_to_concat


def _estimate_memory_of_mosaic(image_paths) -> float | None:
    """
    Estimates the approximate memory allocation (gigabytes) of the full sized
    mosaic prior to cropping.

    Parameters
    ----------
    image_paths: List of final images to combine to get the ideal
                 crop window, generated from _surrounding_img_path_finder.

    Returns
    -------
    est_memory_gb: Float of estimated memory usage in gigabytes for the full mosaic.
                   Returns None if any input file is unreadable or invalid.
    """
    bounds_list = []
    paths_not_fine = []

    for path in image_paths:
        if not _check_raster_file(path):
            paths_not_fine.append(path)

    if paths_not_fine:
        return None

    with rasterio.open(image_paths[0]) as ds:
        res_x, res_y = ds.res
        num_bands = ds.count
        bounds_list.append(ds.bounds)

    for path in image_paths[1:]:
        with rasterio.open(path) as ds:
            bounds_list.append(ds.bounds)

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
    est_memory_gb = est_memory_bytes / (1024**3)

    return est_memory_gb


def _create_new_image(
    image_paths_to_concat: list,
    centre_image_path: str,
    focal_point_easting: float | int,
    focal_point_northing: float | int,
    output_dir: Path,
    file_name: str,
) -> None:
    """
    Function to create the new image with cropped surrounding tiles around
    the user specified focal point.

    Parameters
    ----------
    image_paths_to_concat: List of final images to combine to get the ideal
                           crop window, generated from _surrounding_img_path_finder.
    centre_image_path: String path of the center image which contains the
                       majority of the focal point.
    focal_point_easting: Easting coordinate of focal point.
    focal_point_northing: Northing coordinate of focal point.
    output_dir: Path to output location.
    file_name: Your British National Grid tile that contains the point of
               interest. This name should be taken from the image jpeg name.

    Returns
    -------
    None
    """
    est_mem = _estimate_memory_of_mosaic(image_paths=image_paths_to_concat)
    if est_mem is None:
        LOG.error(
            "Issue with one or all paths required for mosaic. Image expansion not possible."
        )
        return
    if est_mem > 30:
        LOG.error("Could not create mosaic for %s due to memory allocation", centre_image_path)
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
        LOG.error("Error merging mosaic: %s", e)
        return

    if mosaic is None or mosaic.size == 0:
        LOG.error(
            "Mosaic creation failed for %s - empty or null mosaic returned", centre_image_path
        )
        return

    # Convert junction coordinates (in map space) to pixel
    col, row = ~mosaic_transform * (focal_point_easting, focal_point_northing)

    crop_size_pixels = 640
    half_size = crop_size_pixels // 2

    col_start = int(col - half_size)
    row_start = int(row - half_size)
    col_end = int(col + half_size)
    row_end = int(row + half_size)

    if (
        col_start < 0
        or row_start < 0
        or col_end > mosaic.shape[2]
        or row_end > mosaic.shape[1]
    ):
        LOG.warning("%s crop extends beyond mosaic boundaries", centre_image_path)
        LOG.warning("Mosaic shape: %s", mosaic.shape)
        LOG.warning("Crop window: (%s:%s, %s:%s)", row_start, row_end, col_start, col_end)
        return

    # mosaic has shape (bands, height, width)
    cropped_img = mosaic[:, row_start:row_end, col_start:col_end]

    expected_shape = (mosaic.shape[0], crop_size_pixels, crop_size_pixels)
    if cropped_img.shape != expected_shape:
        LOG.warning("%s cropped image has unexpected dimensions", centre_image_path)
        LOG.warning("Expected: %s", expected_shape)
        LOG.warning("Actual: %s", cropped_img.shape)
        return

    # Convert the image: move bands to the last axis and cast to uint8 for JPEG
    final_image = Image.fromarray(np.moveaxis(cropped_img, 0, -1).astype(np.uint8))
    output_filename = os.path.join(output_dir, f"{file_name}_cropped_extended.jpg")
    final_image.save(output_filename, "JPEG", quality=95)

    for ds in datasets:
        ds.close()

    return


def _check_raster_file(path: Path) -> bool:
    """
    Validates whether a raster file can be successfully opened using Rasterio.

    Parameters
    ----------
    path : The path to the raster (image) file to be checked.

    Returns
    -------
    True if the file opens without error; False otherwise.
    """
    try:
        with rasterio.open(path) as _:
            return True
    except RasterioIOError as e:
        print("Error opening file %s: %s", path, e)
        return False
    except RasterioError as e:
        print("Unexpected error with %s: %s", path, e)
        return False

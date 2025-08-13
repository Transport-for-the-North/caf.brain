# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 4/27/2025
Original author: Adil Zaheer
"""
# Built-Ins
import logging
import os
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

# Local Imports
from caf.brain.machine_vision.satellite_image_processing.image_processing.image_processing_functions import (
    check_raster_file,
)

LOG = logging.getLogger(__name__)
# Local Imports
from caf.brain.machine_vision.satellite_image_processing.image_processing.expand_image_functions import (
    create_new_image,
    find_surrounding_images,
    find_surrounding_names,
    surrounding_img_path_finder,
)


def image_crop_main(
    path_list,
    image_supporting_data,
    output,
    image_folder,
    satellite_metadata: pd.DataFrame,
    folder_if_loop: Path,
):
    """
    - must be geo-referenced data

    """
    if folder_if_loop:
        base_name = os.path.basename(folder_if_loop)
        name, _ = os.path.splitext(base_name)
        output_dir = os.path.join(folder_if_loop, f"{name}_images_for_labelling")
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join(output, "images_for_labelling")
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    invalid_count = 0
    invalid_images = []

    with tqdm(total=None) as pbar:
        for path in path_list:
            base_name = os.path.basename(path)
            file_name = os.path.splitext(base_name)[0]
            output_filename = os.path.join(output_dir, f"{file_name}_cropped.jpg")
            output_filename_extended = os.path.join(
                output_dir, f"{file_name}_cropped_extended.jpg"
            )

            if os.path.exists(output_filename) or os.path.exists(output_filename_extended):
                LOG.info(
                    "Skipping image processing as final cropped image already exists in \
                          output directory."
                )
                continue

            row_data = image_supporting_data[
                image_supporting_data["box_boundary"] == file_name
            ]

            if row_data.empty:
                LOG.warning(f"No matching data found for {file_name}")
                pbar.update(1)
                continue

            junction_easting = row_data["coordinates_easting"].values[0]
            junction_northing = row_data["coordinates_northing"].values[0]

            if check_raster_file(path):
                with rasterio.open(path) as img:
                    transform = img.transform
                    # bng to pixels
                    col, row = ~transform * (
                        junction_easting,
                        junction_northing,
                    )  # col and row are pixel coords of target

                    # crop window size in pixels
                    # pixel_resolution = abs(transform[0])  # (find width of pixel in meters)
                    # crop_size_pixels = int(200 / pixel_resolution)
                    crop_size_pixels = 640  # 640 by 640 best for yolo model

                    # half size used so half junc is in the middle of the crop (half one side, half the other)
                    half_size = crop_size_pixels // 2

                    col_start = int(col - half_size)  # left edge of ideal window
                    row_start = int(row - half_size)  # top edge of ideal window
                    col_end = int(col + half_size)  # right edge of ideal window
                    row_end = int(row + half_size)  # bottom edge of ideal window

                    # tells you which tiles are needed for ideal crop window
                    list_of_needed_tiles = find_surrounding_images(
                        col_start=col_start,
                        row_start=row_start,
                        col_end=col_end,
                        row_end=row_end,
                        image_height=img.height,
                        image_width=img.width,
                    )

                    if list_of_needed_tiles:
                        LOG.info(f"Image {path} needs expanding")

                        # gives you the names of all the surrounding tiles
                        image_layout_dict = find_surrounding_names(file_name=file_name)

                        image_paths_to_concat = surrounding_img_path_finder(
                            image_layout_dict=image_layout_dict,
                            list_of_needed_tiles=list_of_needed_tiles,
                            image_folder=image_folder,
                            satellite_metadata=satellite_metadata,
                        )
                        if not image_paths_to_concat:
                            LOG.warning(
                                "Required satellite images to extend centre image for crop \
                                         does not exist"
                            )
                            directory, file_name = os.path.split(path)
                            name = os.path.splitext(file_name)[0]
                            invalid_images.append(name)
                            continue

                        create_new_image(
                            image_paths_to_concat=image_paths_to_concat,
                            centre_image_path=path,
                            junction_easting=junction_easting,
                            junction_northing=junction_northing,
                            output_dir=output_dir,
                            file_name=file_name,
                        )

                        pbar.update(1)
                        count += 1
                        if count % 100 == 0:
                            LOG.info(f"Processed item {count}: {path}")

                    else:
                        LOG.info(f"Image {path} does not need expanding")
                        # Accounting for image boundaries (some junctions are not in the middle of an image)
                        # first two return larger (if col/row_start are negative, then set to 0)
                        # second two return smaller (if col/row_end exceed image bounds, set to actual image bounds)
                        # col_start = max(0, col_start)  # Ensure left edge isn't outside image
                        # row_start = max(0, row_start)  # Ensure top edge isn't outside image
                        # col_end = min(img.width, col_end)  # Ensure right edge doesn't exceed image width
                        # row_end = min(img.height,
                        #               row_end)  # Ensure bottom edge doesn't exceed image height

                        actual_width = col_end - col_start
                        actual_height = row_end - row_start

                        if actual_width <= 0 or actual_height <= 0:
                            LOG.warning(f"Invalid dimensions calculated for {path}")
                            LOG.warning(f"Target point: col={col}, row={row}")
                            LOG.warning(
                                f"Adjusted window: col_start={col_start}, col_end={col_end}, row_start={row_start}, row_end={row_end}"
                            )
                            LOG.warning(
                                f"Image dimensions: width={img.width}, height={img.height}"
                            )

                            directory, file_name = os.path.split(path)
                            name = os.path.splitext(file_name)[0]

                            invalid_images.append(name)

                            invalid_count += 1
                            LOG.info(f"Total invalid images: {invalid_count}")
                            continue

                        window = Window(
                            col_start,  # left edge of box
                            row_start,  # top of box
                            actual_width,  # width
                            actual_height,
                        )  # height

                        # creates 3d array (bands (2d array of pixels, 3 bands for rgb), height, width)
                        cropped_img = img.read(window=window)

                        # create pil image from array with correct format and value order
                        # rearrange array into height width bands format for PIL image
                        final_image = Image.fromarray(
                            np.moveaxis(cropped_img, 0, -1).astype(np.uint8)
                        )
                        final_image.save(output_filename, "JPEG", quality=95)

                        pbar.update(1)
                        count += 1
                        if count % 100 == 0:
                            LOG.info(f"Processed item {count}: {path}")

            else:
                LOG.error(f"Image {path} couldn't be opened")
                directory, file_name = os.path.split(path)
                name = os.path.splitext(file_name)[0]
                invalid_images.append(name)
                continue

    df = pd.DataFrame(invalid_images, columns=["invalid_junctions"])
    df.to_csv(os.path.join(folder_if_loop, "failed_image_crops.csv"), index=False)

    return

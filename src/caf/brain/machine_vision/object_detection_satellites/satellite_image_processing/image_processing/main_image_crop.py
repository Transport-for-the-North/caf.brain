"""
Created on: 4/27/2025
Original author: Adil Zaheer
"""

import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import logging
from pathlib import Path
from caf.brain.machine_vision.object_detection_satellites.satellite_image_processing.image_processing.image_processing_functions import (
    check_raster_file,
)

LOG = logging.getLogger(__name__)
from caf.brain.machine_vision.object_detection_satellites.satellite_image_processing.image_processing.expand_image_functions import (
    find_surrounding_names,
    surrounding_img_path_finder,
    create_new_image,
    find_surrounding_images,
)


def image_crop_main(
    user_coordinate_data: pd.DataFrame,
    output: Path,
    image_folder: Path,
    satellite_metadata: pd.DataFrame,
) -> None:
    """
    Function to locate, crop and where applicable merge images. This function
    creates images ready for labelling or use in a trained YOLO model.

    Parameters
    ----------
    user_coordinate_data: User coordinate data that has been processed by
                          the main_process_user_locations.
    output: Path to output folder.
    image_folder: Path to image folder.
    satellite_metadata: Satellite image metadata created from accompanying
                        xml files and the main_satellite_metadata function.

    Returns
    -------
    None
    """

    base_name = os.path.basename(output)
    name, _ = os.path.splitext(base_name)
    output_dir = os.path.join(output, f"{name}_images_for_labelling")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    invalid_count = 0
    invalid_images = []

    with tqdm(total=None) as pbar:
        for path in user_coordinate_data["paths"]:
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

            row_data = user_coordinate_data[user_coordinate_data["box_boundary"] == file_name]

            if row_data.empty:
                LOG.warning("No matching data found for %s", file_name)
                pbar.update(1)
                continue

            junction_easting = row_data["coordinates_easting"].values[0]
            junction_northing = row_data["coordinates_northing"].values[0]

            if check_raster_file(path):
                with rasterio.open(path) as img:
                    transform = img.transform
                    # bng to pixels, col and row are pixel coords of target
                    col, row = ~transform * (junction_easting, junction_northing)

                    # crop window size in pixels
                    crop_size_pixels = 640

                    # half size so half user coordinates are in the middle of the crop (half one side, half the other)
                    half_size = crop_size_pixels // 2

                    col_start = int(col - half_size)  # left edge of ideal window
                    row_start = int(row - half_size)  # top edge of ideal window
                    col_end = int(col + half_size)  # right edge of ideal window
                    row_end = int(row + half_size)  # bottom edge of ideal window

                    # which tiles are needed for ideal crop window
                    list_of_needed_tiles = find_surrounding_images(
                        col_start=col_start,
                        row_start=row_start,
                        col_end=col_end,
                        row_end=row_end,
                        image_height=img.height,
                        image_width=img.width,
                    )

                    if list_of_needed_tiles:
                        LOG.info("Image %s needs expanding", path)

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
                            LOG.info("Processed item %s: %s", count, path)

                    else:
                        LOG.info("Image %s does not need expanding", path)

                        actual_width = col_end - col_start
                        actual_height = row_end - row_start

                        if actual_width <= 0 or actual_height <= 0:
                            LOG.warning("Invalid dimensions calculated for %s", path)
                            LOG.warning("Target point: col=%s, row=%s", col, row)
                            LOG.warning(
                                "Adjusted window: col_start=%s, col_end=%s, row_start=%s, row_end=%s",
                                col_start,
                                col_end,
                                row_start,
                                row_end,
                            )
                            LOG.warning(
                                "Image dimensions: width=, height=%s", img.width, img.height
                            )

                            directory, file_name = os.path.split(path)
                            name = os.path.splitext(file_name)[0]

                            invalid_images.append(name)

                            invalid_count += 1
                            LOG.info("Total invalid images: %s", invalid_count)
                            continue

                        window = Window(
                            col_start,  # left edge of box
                            row_start,  # top of box
                            actual_width,  # width
                            actual_height,
                        )  # height

                        # 3d array (bands (2d array of pixels, 3 bands for rgb), height, width)
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
    df.to_csv(os.path.join(output_dir, "failed_image_crops.csv"), index=False)

    return

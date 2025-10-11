"""
Created on: 10/6/2025
Original author: Adil Zaheer
"""

# todo also need to check how i did the process of separating out images based on classes
# todo may need to keep that functionality in permanently
import logging
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

from caf.brain.machine_vision.temp_folder.image_creation.crop_user_satellite_images.functions import (
    _check_raster_file,
    _find_surrounding_images,
    _find_surrounding_names,
    _surrounding_img_path_finder,
    create_new_image,
)

LOG = logging.getLogger(__name__)


def image_crop(
    user_image_metadata: pd.DataFrame,
    output_path: Path,
    satellite_image_metadata: pd.DataFrame,
) -> str:
    """
    Function to locate, crop and where applicable merge images. This function
    creates images ready for labelling or use in a trained YOLO model.

    Parameters
    ----------
    user_image_metadata: User coordinate data that has been processed by
                          the main_process_user_locations.
    output_path: Path to output folder.
    satellite_image_metadata: Satellite image metadata containing tile names,
                              midpoints and path locations.

    Returns
    -------
    output_dir: Location of saved processed images
    """
    base_name = os.path.basename(output_path)
    name, _ = os.path.splitext(base_name)
    output_dir = os.path.join(output_path, f"{name}_images_for_machine_vision")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    invalid_count = 0
    invalid_images = []

    with tqdm(total=None) as pbar:
        for path in user_image_metadata["paths"]:
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

            row_data = user_image_metadata[user_image_metadata["box_boundary"] == file_name]

            if row_data.empty:
                LOG.warning("No matching data found for %s", file_name)
                pbar.update(1)
                continue

            focal_point_easting = row_data["coordinates_easting"].values[0]
            focal_point_northing = row_data["coordinates_northing"].values[0]

            if _check_raster_file(path):
                with rasterio.open(path) as img:
                    transform = img.transform
                    # bng to pixels, col and row are pixel coords of target
                    col, row = ~transform * (focal_point_easting, focal_point_northing)

                    # crop window size in pixels
                    crop_size_pixels = 640

                    # half size so half user coordinates are in the middle of the crop (half one side, half the other)
                    half_size = crop_size_pixels // 2

                    col_start = int(col - half_size)
                    row_start = int(row - half_size)
                    col_end = int(col + half_size)
                    row_end = int(row + half_size)

                    # which tiles are needed for ideal crop window
                    list_of_needed_tiles = _find_surrounding_images(
                        col_start=col_start,
                        row_start=row_start,
                        col_end=col_end,
                        row_end=row_end,
                        image_height=img.height,
                        image_width=img.width,
                    )

                    if list_of_needed_tiles:
                        LOG.info("Image %s needs expanding", path)

                        image_layout_dict = _find_surrounding_names(file_name=file_name)

                        image_paths_to_concat = _surrounding_img_path_finder(
                            image_layout_dict=image_layout_dict,
                            list_of_needed_tiles=list_of_needed_tiles,
                            satellite_metadata=satellite_image_metadata,
                        )
                        if not image_paths_to_concat:
                            LOG.warning(
                                "Required satellite images to extend centre image for crop \
                                                 does not exist"
                            )
                            _, file_name = os.path.split(path)
                            name = os.path.splitext(file_name)[0]
                            invalid_images.append(name)
                            continue

                        create_new_image(
                            image_paths_to_concat=image_paths_to_concat,
                            centre_image_path=path,
                            focal_point_easting=focal_point_easting,
                            focal_point_northing=focal_point_northing,
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
                                "Image dimensions: width=%s, height=%s", img.width, img.height
                            )

                            _, file_name = os.path.split(path)
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
                            LOG.info("Processed item %s: %s", count, path)

            else:
                LOG.error("Image %s couldn't be opened", path)
                _, file_name = os.path.split(path)
                name = os.path.splitext(file_name)[0]
                invalid_images.append(name)
                continue

        df = pd.DataFrame(invalid_images, columns=["invalid_junctions"])
        df.to_csv(os.path.join(output_dir, "failed_image_crops.csv"), index=False)

        return output_dir

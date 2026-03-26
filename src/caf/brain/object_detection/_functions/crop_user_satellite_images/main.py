"""
Created on: 10/6/2025
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
from caf.brain.object_detection._functions.crop_user_satellite_images.functions import (
    _check_raster_file,
    _create_new_image,
    _find_surrounding_images,
    _find_surrounding_names,
    _surrounding_img_path_finder,
)

LOG = logging.getLogger(__name__)


def image_crop(
    user_image_metadata: pd.DataFrame,
    output_path: Path,
    satellite_image_metadata: pd.DataFrame,
) -> Path:
    """
    Function to locate, crop and where applicable merge images.

    This function creates images ready for labeling or use in a trained YOLO
    model.

    Parameters
    ----------
    user_image_metadata:
        User coordinate data that has been processed by
        main_process_user_locations.
    output_path:
        Path to output folder.
    satellite_image_metadata:
        Satellite image metadata containing tile names, midpoints and path
        locations. Generated from image_info_generation

    Returns
    -------
    Location of saved processed images.
    """
    output_path = Path(output_path)
    output_dir = output_path / "images_for_machine_vision"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    invalid_count = 0
    invalid_images = []

    with tqdm(total=None) as pbar:
        for idx, row_data in user_image_metadata.iterrows():
            path = row_data["paths"]
            file_name = row_data["box_boundary"]

            if "id" in row_data and pd.notna(row_data["id"]):
                unique_id = str(row_data["id"])
            else:
                unique_id = f"{file_name}_{idx}"

            output_filename = output_dir / f"{unique_id}_cropped.jpg"
            output_filename_extended = output_dir / f"{unique_id}_cropped_extended.jpg"

            if output_filename.exists() or output_filename_extended.exists():
                LOG.info("Skipping image processing as final cropped image already exists")
                pbar.update(1)
                continue

            focal_point_easting = row_data["coordinates_easting"]
            focal_point_northing = row_data["coordinates_northing"]

            if _check_raster_file(path):
                with rasterio.open(path) as img:
                    transform = img.transform
                    col, row = ~transform * (focal_point_easting, focal_point_northing)
                    crop_size_pixels = 640
                    half_size = crop_size_pixels // 2

                    col_start = int(col - half_size)
                    row_start = int(row - half_size)
                    col_end = int(col + half_size)
                    row_end = int(row + half_size)

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

                        _create_new_image(
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

                        cropped_img = img.read(window=window)

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
        df.to_csv(output_dir / "failed_image_crops.csv", index=False)

        return output_dir

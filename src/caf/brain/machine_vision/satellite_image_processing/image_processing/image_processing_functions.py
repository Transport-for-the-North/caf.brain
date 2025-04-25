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
import rasterio
from rasterio.windows import Window
from PIL import Image
from pathlib import Path
from rasterio.merge import merge
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


def image_crop(path_list,
               image_supporting_data,
               output,
               image_folder):
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
            print("@@@@@@@@path@@@@@@@@@")
            print(path)
            base_name = os.path.basename(path)
            file_name = os.path.splitext(base_name)[0]

            row_data = image_supporting_data[image_supporting_data['box_boundary'] == file_name]

            if row_data.empty:
                LOG.warning(f"No matching data found for {file_name}")
                pbar.update(1)
                continue

            junction_easting = row_data['coordinates_easting'].values[0]
            junction_northing = row_data['coordinates_northing'].values[0]

            with rasterio.open(path) as img:
                transform = img.transform
                # bng to pixels
                col, row = ~transform * (junction_easting, junction_northing)  # col and row are pixel coords of target

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

                need_to_expand_image = []

                if col_start >= 0:
                    need_to_expand_image.append("missing_left")
                elif row_start >= 0:
                    need_to_expand_image.append("missing_top")
                elif col_end <= img.image_width:
                    need_to_expand_image.append("missing_right")
                elif row_end <= img.image_height:
                    need_to_expand_image.append("missing_bottom")


                # todo move find_surroudning_names and find_surroudning_images outside of if
                # todo use them to detemine the if


                if need_to_expand_image:
                    LOG.info(f"Image {path} needs expanding")
                    print(need_to_expand_image)
                    # gives you the names of all the surrounding tiles
                    image_layout_dict = find_surrounding_names(file_name=file_name)
                    print("image_layout_dict")
                    print(image_layout_dict)
                    # tells you which tiles are needed for ideal crop window
                    list_of_needed_tiles = find_surrounding_images(col_start=col_start,
                                                                   row_start=row_start,
                                                                   col_end=col_end,
                                                                   row_end=row_end,
                                                                   image_height=img.height,
                                                                   image_width=img.width)
                    print("list_of_needed_tiles")
                    print(list_of_needed_tiles)
                    image_paths_to_concat = surrounding_img_path_finder(image_layout_dict=image_layout_dict,
                                                                        list_of_needed_tiles=list_of_needed_tiles,
                                                                        image_folder=image_folder)
                    print("image_paths_to_concat")
                    print(image_paths_to_concat)
                    create_new_image(image_paths_to_concat=image_paths_to_concat,
                                     centre_image_path=path,
                                     junction_easting=junction_easting,
                                     junction_northing=junction_northing,
                                     output_dir=output_dir,
                                     file_name=file_name)

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
                        LOG.warning(f"Adjusted window: col_start={col_start}, col_end={col_end}, row_start={row_start}, row_end={row_end}")
                        LOG.warning(f"Image dimensions: width={img.width}, height={img.height}")

                        directory, file_name = os.path.split(path)
                        name = os.path.splitext(file_name)[0]

                        invalid_images.append(name)

                        invalid_count += 1
                        LOG.info(f'Total invalid images: {invalid_count}')
                        continue

                    window = Window(col_start,  # left edge of box
                                    row_start,  # top of box
                                    actual_width,  # width
                                    actual_height)  # height

                    # creates 3d array (bands (2d array of pixels, 3 bands for rgb), height, width)
                    cropped_img = img.read(window=window)

                    # create pil image from array with correct format and value order
                    # rearrange array into height width bands format for PIL image
                    final_image = Image.fromarray(np.moveaxis(cropped_img, 0, -1).astype(np.uint8))
                    print(final_image)
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
                       image_reference_column: str,
                       output: Path) -> list[Path]:

    path_file_name = "satellite_image_paths.csv"
    if os.path.exists(os.path.join(output, path_file_name)):
        df = pd.read_csv(os.path.join(output, path_file_name))
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
        path_list_df.to_csv(os.path.join(output, path_file_name), index=False)

    return path_list


def find_surrounding_names(file_name) -> dict:
    file_name_list = list(file_name)
    prefix = file_name_list[:2]
    prefix = "".join(prefix)

    nums = file_name_list[2:]
    concat_nums = ''.join(nums)
    nums_int = int(concat_nums)

    east = nums_int + 100
    west = nums_int - 100
    north = nums_int + 1
    south = nums_int - 1
    northeast = nums_int + 101
    northwest = nums_int - 99
    southeast = nums_int + 99
    southwest = nums_int - 101

    image_layout_dict = {'centre_image': file_name,
                         'north': north,
                         'south': south,
                         'east': east,
                         'west': west,
                         'northeast': northeast,
                         'southeast': southeast,
                         'southwest': southwest,
                         'northwest': northwest}

    for key in image_layout_dict:
        image_layout_dict[key] = prefix + str(image_layout_dict[key])

    return image_layout_dict


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


def surrounding_img_path_finder(image_layout_dict,
                                list_of_needed_tiles,
                                image_folder):

    images_to_concat = []

    for key, val in image_layout_dict.items():
        for item in list_of_needed_tiles:
            if item == key:
                images_to_concat.append(val)

    paths = glob.glob(os.path.join(image_folder) + '/**/*.jpg', recursive=True)

    for path in paths:
        base_name = os.path.basename(path)
        file_name = os.path.splitext(base_name)[0]

        for image in images_to_concat:
            if image == file_name:
                images_to_concat[image] = path

    return images_to_concat


def create_new_image(image_paths_to_concat,
                     centre_image_path,
                     junction_easting,
                     junction_northing,
                     output_dir,
                     file_name):
    datasets = []
    for path in image_paths_to_concat:
        ds = rasterio.open(path)
        datasets.append(ds)

    centre_ds = rasterio.open(centre_image_path)
    datasets.append(centre_ds)

    # Merge datasets. returns NumPy array (mosaic) and the transform
    mosaic, mosaic_transform = merge(datasets)

    # Convert junction coordinates (in map space) to pixel
    col, row = ~mosaic_transform * (junction_easting, junction_northing)

    crop_size_pixels = 640
    half_size = crop_size_pixels // 2

    col_start = int(col - half_size)
    row_start = int(row - half_size)
    col_end = int(col + half_size)
    row_end = int(row + half_size)

    # Calculate width and height for the window
    # width = col_end - col_start
    # height = row_end - row_start
    # window = Window(col_start, row_start, width, height)

    # mosaic has shape (bands, height, width)
    cropped_img = mosaic[:, row_start:row_end, col_start:col_end]

    # Convert the image: move bands to the last axis and cast to uint8 for JPEG
    final_image = Image.fromarray(np.moveaxis(cropped_img, 0, -1).astype(np.uint8))

    output_filename = os.path.join(output_dir, f"{file_name}_cropped.jpg")
    final_image.save(output_filename, "JPEG", quality=95)

    for ds in datasets:
        ds.close()

    return

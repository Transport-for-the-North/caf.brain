"""
Created on: 7/22/2025
Original author: Adil Zaheer
"""
import os.path
import logging
import pandas as pd
from pathlib import Path
import glob
from caf.brain.machine_vision.object_detection_satellites.satellite_image_processing.image_processing.image_processing_functions import \
    euclidean_distance
import geopandas as gpd
from tqdm import tqdm
LOG = logging.getLogger(__name__)


def main_process_user_locations(user_locations_path: Path,
                                output_path: Path,
                                satellite_metadata: pd.DataFrame,
                                image_folder: Path,
                                x_coordinate: str = None,
                                y_coordinate: str = None,
                                ) -> pd.DataFrame:
    """
    Function to locate the satellite images that correspond to the user input
    coordinates.

    Parameters
    ----------
    user_locations_path: Path of user data that must contain coordinates in
                         some form.
    output_path: Path to output folder.
    satellite_metadata: Satellite images metadata created from main_satellite_metadata.
    image_folder: Path to images.
    x_coordinate: X coordinate column inside your data if data is not a
                  shapefile.
    y_coordinate: Y coordinate column inside your data if data is not a
                  shapefile.

    Returns
    -------
    final_data: final processed input data to be used for image generation.

    """
    user_locations_path = os.path.join(user_locations_path)
    user_coordinate_filename = "user_coordinate_data.csv"

    if os.path.exists(os.path.join(output_path, user_coordinate_filename)):
        df = pd.read_csv(os.path.join(output_path, user_coordinate_filename))
        return df

    df, file_type = process_file(file_path=user_locations_path)
    if file_type == "shp":
        if 'geometry' not in df.columns:
            raise ValueError("Ensure that geometry column is present in Geography data.")

        df['coordinates_easting'] = df['geometry'].x
        df['coordinates_northing'] = df['geometry'].y
    else:
        df = df.rename(columns={x_coordinate: 'coordinates_easting',
                                y_coordinate: 'coordinates_northing'})

    coordinate_data, _ = euclidean_distance(df_a=satellite_metadata,
                                            df_b=df)

    path_df = user_path_finder(image_folder=image_folder,
                               user_coordinate_data=coordinate_data)

    final_data = pd.merge(coordinate_data, path_df, on='box_boundary', how='inner')

    final_data.to_csv(os.path.join(output_path, user_coordinate_filename))

    return final_data


def get_file_type(file_path: Path) -> str:
    """
    Helper function for finding file types.

    Parameters
    ----------
    file_path: Path to file.

    Returns
    -------
    String with file ending e.g. csv, shp
    """

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.csv':
        return 'csv'
    elif ext == '.shp':
        return 'shp'
    else:
        return None


def process_file(file_path: Path):
    """
    Function to read in files based on their file types.

    Parameters
    ----------
    file_path: Path to file.

    Returns
    -------
    Either a dataframe or geo-dataframe of your data along with their file
    type.

    """
    file_type = get_file_type(file_path)
    if file_type == 'csv':
        df = pd.read_csv(file_path)
        return df, file_type
    elif file_type == 'shp':
        gdf = gpd.read_file(file_path)
        return gdf, file_type
    else:
        raise ValueError("Unsupported file type.")


def user_path_finder(image_folder: Path,
                     user_coordinate_data,
                       ) -> list[Path]:
    """
    User to find image paths for each of the user provided coordinates.

    Parameters
    ----------
    image_folder: Path to image folder.
    user_coordinate_data: User coordinate data after euclidean_distance
                          function use.

    Returns
    -------
    path_df: Satellite image path dataframe that corresponds to the user
             provided locations.
    """

    counter = 0
    paths = glob.glob(os.path.join(image_folder) + '/**/*.jpg', recursive=True)

    user_coordinate_data['box_boundary'] = user_coordinate_data['box_boundary'].astype(str)
    reference_set = set(user_coordinate_data['box_boundary'].values)

    path_dict = {}
    LOG.info("Looking for %s unique reference values in %s image files", len(reference_set), len(paths))

    with tqdm(total=None) as pbar:
        for path in paths:
            _, file_name = os.path.split(path)
            name = os.path.splitext(file_name)[0]

            if name in reference_set:
                path_dict[name] = path
            pbar.update(1)
            counter += 1
            if counter % 100 == 0:
                LOG.info("Processed %s paths, found %s matches so far", (counter/len(paths)), len(path_list))

    path_df = pd.DataFrame(list(path_dict.items()), columns=['box_boundary', 'paths'])

    return path_df

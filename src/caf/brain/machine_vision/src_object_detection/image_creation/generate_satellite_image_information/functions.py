"""
Created on: 9/17/2025
Original author: Adil Zaheer
"""

from pathlib import Path
import re
import logging
import os
import glob
import pandas as pd
from bs4 import BeautifulSoup
from pyproj import Transformer
from tqdm import tqdm

LOG = logging.getLogger(__name__)


def _TEMP_satellite_xml_processor(image_folder_path_north: Path, output_path: Path,
                                  image_folder_path_south: Path) -> pd.DataFrame:
    """
    Creates a dataframe of British National Grid satellite image xml data.

    ***************
    THIS IS TEMPORARY UNTIL WE ARE ABLE TO COMBINE NORTH AND SOUTH
    BLUESKY IMAGES
    ***************

    Parameters
    ----------
    image_folder_path: Path to folder that contains xml files.
    output_path: Path to output location.

    Returns
    -------
    final_data: Dataframe of xml information for evey file in the folder_path.
    """
    LOG.info("XML processing beginning")
    problem_html = []

    data_dict = {}
    counter = 0
    paths_north = glob.glob(os.path.join(image_folder_path_north) + "/**/*.xml", recursive=True)
    paths_south = glob.glob(os.path.join(image_folder_path_south) + "/**/*.xml", recursive=True)

    with tqdm(total=None) as pbar:
        for path in paths_north + paths_south:
            df, name = _extract_info_from_xml(html_file_path=path)
            if df is None or df.empty:
                LOG.warning("Skipping file %s due to missing or empty data.", path)
                problem_html.append(path)
                continue
            data_dict[name] = df
            pbar.update(1)
            counter += 1
            if counter % 1000 == 0:
                LOG.info("Processed item %s: %s", counter, path)

    final_data = pd.concat(data_dict.values(), ignore_index=True)

    problem_df = pd.DataFrame(problem_html)
    problem_df.to_csv(os.path.join(output_path, "problem_xml_files.csv"), index=False)

    LOG.info("XML processing ending")
    return final_data


def _satellite_xml_processor(image_folder_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Creates a dataframe of British National Grid satellite image xml data.

    Parameters
    ----------
    image_folder_path: Path to folder that contains xml files.
    output_path: Path to output location.

    Returns
    -------
    final_data: Dataframe of xml information for evey file in the folder_path.
    """
    LOG.info("XML processing beginning")
    problem_html = []

    data_dict = {}
    counter = 0
    paths = glob.glob(os.path.join(image_folder_path) + "/**/*.xml", recursive=True)
    with tqdm(total=None) as pbar:
        for path in paths:
            df, name = _extract_info_from_xml(html_file_path=path)
            if df is None or df.empty:
                LOG.warning("Skipping file %s due to missing or empty data.", path)
                problem_html.append(path)
                continue
            data_dict[name] = df
            pbar.update(1)
            counter += 1
            if counter % 1000 == 0:
                LOG.info("Processed item %s: %s", counter, path)

    final_data = pd.concat(data_dict.values(), ignore_index=True)

    problem_df = pd.DataFrame(problem_html)
    problem_df.to_csv(os.path.join(output_path, "problem_xml_files.csv"), index=False)

    LOG.info("XML processing ending")
    return final_data


def _extract_info_from_xml(html_file_path: str) -> tuple:
    """
    Extracting key information from British National Grid xml files.

    Parameters
    ----------
    html_file_path: path to image xml file.

    Returns
    -------
    df: dataframe of xml data information for a specific British National Grid image tile.
    location: string location of the processed British National Grid image tile.
    """
    df = None
    location = None
    with open(html_file_path, "r", encoding="utf-8") as file:
        xml_data = file.read()

    soup = BeautifulSoup(xml_data, "lxml-xml")

    tile = soup.find("gmd:supplementalInformation")
    if tile is not None:
        string = tile.find("gco:CharacterString")
        if string is not None:
            final_string = str(string.text)

            west = float(soup.find("gmd:westBoundLongitude").find("gco:Decimal").text)
            east = float(soup.find("gmd:eastBoundLongitude").find("gco:Decimal").text)
            south = float(soup.find("gmd:southBoundLatitude").find("gco:Decimal").text)
            north = float(soup.find("gmd:northBoundLatitude").find("gco:Decimal").text)

            df = pd.DataFrame(
                {
                    "bng_location": final_string,
                    "box_boundary": ["West", "East", "South", "North"],
                    "wgs84_lat_long_coordinate": [west, east, south, north],
                }
            )

            string = df.at[0, "bng_location"]
            substrings = re.findall(r"\w+", string)
            location = substrings[2]
            df["bng_location"] = location

            df = _find_midpoint(df, location)

        else:
            LOG.warning(
                "Box boundary string was not found in the xml file for %s.", html_file_path
            )

    else:
        LOG.warning(
            "Box boundary string was not found in the xml file for %s.", html_file_path
        )

    return df, location


def _find_midpoint(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Finding the midpoint of a British National Grid image tile.

    Parameters
    ----------
    df: British National Grid tile dataframe based on xml data
    location: string of which British National Grid tile location is being processed.

    Returns
    -------
    result_df: DataFrame of a British National Grid tile metadata including
               midpoint coordinates in GCS and National Grid formats.
    """
    df_1 = df.copy()

    northing = df_1[df_1["box_boundary"].isin(["South", "North"])]
    easting = df_1[df_1["box_boundary"].isin(["West", "East"])]

    lat = northing["wgs84_lat_long_coordinate"].mean()
    long = easting["wgs84_lat_long_coordinate"].mean()

    midpoint_df = pd.DataFrame(
        {
            "box_boundary": location,
            "tile_latitude_wgs84": [lat],
            "tile_longitude_wgs84": [long],
        }
    )

    result_df = _coordinate_converter(df=midpoint_df)

    return result_df


def _coordinate_converter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts coordinates between systems (latitude & longitude to easting &
    northing).

    Parameters
    ----------
    df: British National Grid midpoint tile location.

    Returns
    -------
    final_df: Input dataframe with converted coordinates attached.
    """
    # long / lat -> easting / northing
    df_ = df.copy()

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    mid = (df_["tile_longitude_wgs84"][0], df_["tile_latitude_wgs84"][0])

    mid_bng = transformer.transform(*mid)
    mid_bng = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, mid_bng))

    easting_northing_df = pd.DataFrame(
        {"tile_easting": [mid_bng[0]], "tile_northing": [mid_bng[1]]}
    )

    final_df = pd.concat([df, easting_northing_df], axis=1)

    return final_df


def _image_path_name_finder(folder_path: Path) -> pd.DataFrame:
    """
    Helper function for finding satellite image names and path locations.

    Parameters
    ----------
    folder_path: Path to folder that contains the satellite images (JPG).

    Returns
    -------
    df: Dataframe of image paths and their box boundary (BNG name).
    """
    meta_dict = {}

    paths = glob.glob(os.path.join(folder_path) + "/**/*.jpg", recursive=True)
    for path in paths:
        _, file_name = os.path.split(path)
        name = os.path.splitext(file_name)[0]
        meta_dict[name] = path

    df = pd.DataFrame(list(meta_dict.items()), columns=["box_boundary", "path"])

    return df


def _TEMP_image_path_name_finder(image_folder_path_north: Path,
                                 image_folder_path_south: Path) -> pd.DataFrame:
    """
    Helper function for finding satellite image names and path locations.

    ***************
    THIS IS TEMPORARY UNTIL WE ARE ABLE TO COMBINE NORTH AND SOUTH
    BLUESKY IMAGES
    ***************

    Parameters
    ----------
    folder_path: Path to folder that contains the satellite images (JPG).

    Returns
    -------
    df: Dataframe of image paths and their box boundary (BNG name).
    """
    meta_dict = {}

    paths_north = glob.glob(os.path.join(image_folder_path_north) + "/**/*.jpg", recursive=True)
    paths_south = glob.glob(os.path.join(image_folder_path_south) + "/**/*.jpg", recursive=True)

    for path in paths_north + paths_south:
        _, file_name = os.path.split(path)
        name = os.path.splitext(file_name)[0]
        meta_dict[name] = path

    df = pd.DataFrame(list(meta_dict.items()), columns=["box_boundary", "path"])

    return df

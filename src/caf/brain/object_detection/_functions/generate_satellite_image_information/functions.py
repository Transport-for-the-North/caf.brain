"""
Created on: 9/17/2025
Original author: Adil Zaheer
"""

# Built-Ins
import glob
import logging
import os
import re
from pathlib import Path

# Third Party
import pandas as pd
from bs4 import BeautifulSoup
from pyproj import Transformer
from tqdm import tqdm

LOG = logging.getLogger(__name__)


def _satellite_xml_processor(image_folder_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Creates dataframe of British National Grid satellite image XML data.

    Parameters
    ----------
    image_folder_path:
        Path to folder that contains XML files.
    output_path:
        Path to output location.

    Returns
    -------
    final_data:
        Dataframe of XML information for evey file in the folder_path.
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
    Extracting key information from British National Grid XML files.

    Parameters
    ----------
    html_file_path:
        Path to image XML file.

    Returns
    -------
    df:
        Dataframe of XML data information for a specific British National Grid
        image tile.
    location:
        String location of the processed British National Grid image tile.
    """
    df = None
    location = None
    with open(html_file_path, "r", encoding="utf-8") as file:
        xml_data = file.read()

    soup = BeautifulSoup(xml_data, "lxml-xml")

    def _get_decimal(tag_name: str) -> float | None:
        """
        Helper function to extract image info.

        Parameters
        ----------
        tag_name:
            Beautiful Soup tile tag.

        Returns
        -------
        Box boundary where applicable.
        """
        tag = soup.find(tag_name)
        if tag is None:
            return None
        dec = tag.find("gco:Decimal")
        if dec is None or dec.text is None:
            return None
        try:
            return float(dec.text)
        except (ValueError, TypeError):
            return None

    tile = soup.find("gmd:supplementalInformation")
    if tile is not None:
        string = tile.find("gco:CharacterString")
        if string is not None:
            final_string = str(string.text)

            west = _get_decimal("gmd:westBoundLongitude")
            east = _get_decimal("gmd:eastBoundLongitude")
            south = _get_decimal("gmd:southBoundLatitude")
            north = _get_decimal("gmd:northBoundLatitude")

            if None in (west, east, south, north):
                LOG.warning(
                    "Box boundary string was not found in the xml file for %s.", html_file_path
                )
                return df, location

            df = pd.DataFrame(
                {
                    "bng_location": final_string,
                    "box_boundary": ["West", "East", "South", "North"],
                    "wgs84_lat_long_coordinate": [west, east, south, north],
                }
            )

            raw_string = str(df.at[0, "bng_location"])
            substrings = re.findall(r"\w+", raw_string)
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
    df:
        Dataframe of XML data information for a specific British National Grid
        image tile.
    location:
        String location of the processed British National Grid image tile.
    Returns
    -------
    result_df:
        DataFrame of a British National Grid tiles metadata including midpoint
        coordinates in GCS and National Grid formats.
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
    Converts coordinates systems (latitude & longitude to easting & northing).

    long / lat -> easting / northing

    Parameters
    ----------
    df:
        British National Grid midpoint tile location.

    Returns
    -------
    final_df:
        Input dataframe with converted coordinates attached.
    """
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
    folder_path:
        Path to folder that contains the satellite images (JPG).

    Returns
    -------
    df:
        Dataframe of image paths and their box boundary (BNG name).
    """
    meta_dict = {}

    paths = glob.glob(os.path.join(folder_path) + "/**/*.jpg", recursive=True)
    for path in paths:
        _, file_name = os.path.split(path)
        name = os.path.splitext(file_name)[0]
        meta_dict[name] = path

    df = pd.DataFrame(list(meta_dict.items()), columns=["box_boundary", "path"])

    return df

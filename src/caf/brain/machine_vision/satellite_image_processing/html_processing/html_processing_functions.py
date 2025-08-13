# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 3/3/2025
Original author: Adil Zaheer
"""
# Built-Ins
import logging
import re
from pathlib import Path

# Third Party
import pandas as pd
from bs4 import BeautifulSoup
from pyproj import Transformer

LOG = logging.getLogger(__name__)


def extract_info_from_html(html_file_path: Path) -> tuple:
    """

    :param html_file_path:

    :return:
    """
    df = None
    location = None
    data = []
    with open(html_file_path, "r", encoding="utf-8") as file:
        xml_data = file.read()

    soup = BeautifulSoup(xml_data, "lxml-xml")

    tile = soup.find("gmd:supplementalInformation")
    if tile is not None:
        string = tile.find("gco:CharacterString")
        if string is not None:
            final_string = str(string.text)

            # tile = str(soup.find('gmd:supplementalInformation').find('gco:CharacterString').text)
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
            pattern = r"\w+"
            substrings = re.findall(pattern, string)
            location = substrings[2]
            df["bng_location"] = location

            df = find_midpoint(df, location)

            data.append(df)

        else:
            LOG.warning(
                f"Box boundary string was not found in the xml file for \
                          {html_file_path}."
            )

    else:
        LOG.warning(
            f"Box boundary string was not found in the xml file for \
                      {html_file_path}."
        )

    return df, location


def find_midpoint(df, location):
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

    result_df = coordinate_converter(df=midpoint_df)

    return result_df


def coordinate_converter(df):
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

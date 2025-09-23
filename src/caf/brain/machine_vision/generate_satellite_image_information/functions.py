"""
Created on: 9/17/2025
Original author: Adil Zaheer
"""


import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import re
from pyproj import Transformer
import os
import glob
from tqdm import tqdm
import logging
LOG = logging.getLogger(__name__)


def satellite_xml_processor(folder_path: Path,
                            output_path: Path) -> pd.DataFrame:
    """
    Creates a dataframe of satellite image xml data.

    Parameters
    ----------
    folder_path: Path to folder that contains xml files.
    output_path: Path to output location.

    Returns
    -------
    final_data: Dataframe of xml information for evey file in the folder_path.
    """
    LOG.info('XML processing beginning')
    problem_html = []

    data_dict = {}
    counter = 0
    paths = glob.glob(os.path.join(folder_path) + '/**/*.xml', recursive=True)
    with tqdm(total=None) as pbar:
        for path in paths:
            df, name = extract_info_from_html(html_file_path=path)
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

    LOG.info('XML processing ending')
    return final_data


def extract_info_from_html(html_file_path: Path) -> tuple:
    """

    :param html_file_path:

    :return:
    """
    df = None
    location = None
    data = []
    with open(html_file_path, 'r', encoding='utf-8') as file:
        xml_data = file.read()

    soup = BeautifulSoup(xml_data, 'lxml-xml')

    tile = soup.find('gmd:supplementalInformation')
    if tile is not None:
        string = tile.find('gco:CharacterString')
        if string is not None:
            final_string = str(string.text)

            # tile = str(soup.find('gmd:supplementalInformation').find('gco:CharacterString').text)
            west = float(soup.find('gmd:westBoundLongitude').find('gco:Decimal').text)
            east = float(soup.find('gmd:eastBoundLongitude').find('gco:Decimal').text)
            south = float(soup.find('gmd:southBoundLatitude').find('gco:Decimal').text)
            north = float(soup.find('gmd:northBoundLatitude').find('gco:Decimal').text)

            df = pd.DataFrame({
                'bng_location': final_string,
                'box_boundary': ['West', 'East', 'South', 'North'],
                'wgs84_lat_long_coordinate': [west, east, south, north]
            })

            string = df.at[0, 'bng_location']
            pattern = r'\w+'
            substrings = re.findall(pattern, string)
            location = substrings[2]
            df['bng_location'] = location

            df = find_midpoint(df, location)

            data.append(df)

        else:
            LOG.warning(f"Box boundary string was not found in the xml file for \
                          {html_file_path}.")

    else:
        LOG.warning(f"Box boundary string was not found in the xml file for \
                      {html_file_path}.")

    return df, location


def find_midpoint(df, location):
    df_1 = df.copy()

    northing = df_1[df_1['box_boundary'].isin(['South', 'North'])]
    easting = df_1[df_1['box_boundary'].isin(['West', 'East'])]

    lat = northing['wgs84_lat_long_coordinate'].mean()
    long = easting['wgs84_lat_long_coordinate'].mean()

    midpoint_df = pd.DataFrame({'box_boundary': location,
                                'tile_latitude_wgs84': [lat],
                                'tile_longitude_wgs84': [long]})

    result_df = coordinate_converter(df=midpoint_df)

    return result_df


def coordinate_converter(df):
    # long / lat -> easting / northing
    df_ = df.copy()

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    mid = (df_['tile_longitude_wgs84'][0], df_['tile_latitude_wgs84'][0])

    mid_bng = transformer.transform(*mid)
    mid_bng = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, mid_bng))

    easting_northing_df = pd.DataFrame({'tile_easting': [mid_bng[0]],
                                        'tile_northing': [mid_bng[1]]
                                        })

    final_df = pd.concat([df, easting_northing_df], axis=1)

    return final_df


def image_path_name_finder(folder_path: Path) -> pd.DataFrame:
    """
    Helper function for finding satellite image names and path locations.

    Parameters
    ----------
    folder_path: Path to folder that contains the satelite images (JPG).

    Returns
    -------
    df: Dataframe of image paths and their box boundary (BNG name).
    """
    meta_dict = {}

    paths = glob.glob(os.path.join(folder_path) + '/**/*.jpg', recursive=True)
    for path in paths:
        directory, file_name = os.path.split(path)
        name = os.path.splitext(file_name)[0]
        meta_dict[name] = path

    df = pd.DataFrame(list(meta_dict.items()), columns=['box_boundary', 'path'])

    return df

# -*- coding: utf-8 -*-
"""
Created on: 3/10/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os
import pandas as pd
import geopandas as gpd
from dbfread import DBF
from pathlib import Path
from caf.brain.machine_vision.machine_vision_inputs import NoHAMInputs


def noham_data(geo_path: Path,
               noham_path: Path,
               output: Path) -> pd.DataFrame:
    df_geo = gpd.read_file(geo_path)
    table = DBF(noham_path)
    records = [record for record in table]
    df = pd.DataFrame(records)

    df_geo = df_geo[['N', 'JNCTYPE', 'geometry']]
    df_geo = df_geo[df_geo.N > 10000]
    df_geo = df_geo.dropna()
    index_vals = df_geo[df_geo['JNCTYPE'] == 'EX'].index
    df_geo = df_geo.drop(index_vals).reset_index(drop=True)
    df_geo.loc[df_geo["JNCTYPE"] == "p", "JNCTYPE"] = 'P'
    df_geo.loc[df_geo["JNCTYPE"] == "s", "JNCTYPE"] = 'S'

    df = df[df.B > 10000].reset_index(drop=True)
    df = df[['B', 'RDCLASS', 'MIDLANE', 'STPLANE', 'MAJOR']]
    df = df.rename(columns={'B': 'N'})

    index_val = df[df['RDCLASS'] == 'ZC'].index
    df = df.drop(index_val).reset_index(drop=True)

    index_val = df[df['RDCLASS'] == 'zc'].index
    df = df.drop(index_val).reset_index(drop=True)

    df.loc[df["RDCLASS"].isin(["A Road", "A", "A ROAD"]), "RDCLASS"] = 'a_road'
    df.loc[df["RDCLASS"].isin(["B Road", "B", "B ROA"]), "RDCLASS"] = 'b_road'
    df.loc[df["RDCLASS"].isin(["Motorway", "MOTORWAY", "M"]), "RDCLASS"] = 'motorway'
    df.loc[df["RDCLASS"].isin(["U", "Unclassified"]), "RDCLASS"] = 'unclassified'
    df.loc[df["RDCLASS"].isin(["Minor Road", "MINOR ROAD", "Minor", "MINOR"]), "RDCLASS"] = 'minor_road'
    df.loc[df["MAJOR"] == "Y", "MAJOR"] = 'y'
    df.loc[df["MAJOR"] == "N", "MAJOR"] = 'n'
    df.loc[df["MAJOR"] == "", "MAJOR"] = 'empty'
    df.loc[df["RDCLASS"] == "", "RDCLASS"] = 'empty'

    noham_data = df_geo.merge(df, how='outer', on='N')
    noham_data.dropna(subset=['geometry'], inplace=True)
    noham_data.fillna(value='was_nan_value', inplace=True)
    noham_data['coordinates_easting'] = noham_data['geometry'].x
    noham_data['coordinates_northing'] = noham_data['geometry'].y

    noham_data.reset_index(drop=True, inplace=True)
    noham_data.to_csv(os.path.join(output, 'noham_data.csv'), index=False)

    return noham_data


def main_(parmas: NoHAMInputs):
    noham_dat = noham_data(geo_path=parmas.noham_shp_path,
                           noham_path=params.noham_db_path,
                           output=parmas.output_path)
    return noham_dat

# larger db smaller shp?
if __name__ == "__main__":
    params = NoHAMInputs(
        output_path=Path(r"E:\2025 work streams\caf.brAIn\machine vision\MVP work\input_image_processing"),
        noham_db_path=Path(r"E:\2025 work streams\caf.brAIn\machine vision\MVP work\input_image_processing\input\Base_2018_shapefiles\NoHAM_Base.DBF"),
        noham_shp_path=Path(r"E:\2025 work streams\caf.brAIn\machine vision\MVP work\input_image_processing\input\Base_2018_shapefiles\NoHAM_Base_node.shp")
    )
    main_(params)

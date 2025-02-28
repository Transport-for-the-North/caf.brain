# -*- coding: utf-8 -*-
"""
Created on: 2/18/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position

import pandas as pd
from dbfread import DBF

path = r"E:\2025 work streams\caf.brAIn\machine vision\NN input data\04.Final Base Network\NoHAM_Base.DBF"
table = DBF(path)
records = [record for record in table]
df = pd.DataFrame(records)
print(df)

import geopandas as gpd

path_geo = r"E:\2025 work streams\caf.brAIn\machine vision\NN input data\shapefiles"
df_geo = gpd.read_file(path_geo)

# test = gpd.GeoDataFrame(test, geometry='column name')



def transform_network_points():
    """
    Function to transform coordinates from British National Grid format to
    worldwide latitude and longitude.
    """
    # first call caf.ml class to tidy data
    # convert all points using to_crs
    # output needs to be something in the format that can then be fed into getting map and satellite data
    # in the end users will provide maybe just a set of \
    # coordinates so doesn't need to go in the class, or they draw on a map \
    # those would be inputs
    # add this step to project plan
    # document as you go, doc strings too

    return

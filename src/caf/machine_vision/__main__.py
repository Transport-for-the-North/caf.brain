# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 2/18/2025
Original author: Adil Zaheer
"""
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

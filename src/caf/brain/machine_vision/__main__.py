# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 2/18/2025
Original author: Adil Zaheer
"""
import time
import os
import pandas as pd
from caf.brain.machine_vision.html_processing.html_processing_main import main_process_html
from caf.brain.machine_vision.image_processing.process_images_main import main_process_images
from caf.brain.machine_vision.noham_processing.noham_processing_main import main_process_noham
import logging
LOG = logging.getLogger(__name__)

def main(params):
    start_time = time.time()

    output_path = os.path.join(params.output_path, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # html_data = main_process_html(folder_path=params.training_data,
    #                               output_path=output_path)
    #
    # noham_data = main_process_noham(output_path=output_path,
    #                                 noham_path=params.noham_path,
    #                                 geo_path=params.geo_path)

    path = r"E:\2025 work streams\caf.brAIn\machine vision\output\final_html_data.csv"
    path1 = r"E:\2025 work streams\caf.brAIn\machine vision\output\noham_data.csv"

    html_data = pd.read_csv(path)
    noham_data = pd.read_csv(path1)

    main_process_images(html_data=html_data,
                        noham_data=noham_data,
                        output=output_path,
                        training_data_path=params.training_data)

    end_time = time.time()
    LOG.info(f"Total run time: {end_time - start_time:.2f} seconds")

    return



# todo read_and_store_images only needs to be ones where we have junctions? runs out of memory otherwise
# images_dict = read_and_store_images(folder_path=folder_path,
#                                     output_path=output_path)










# model_path = Path(r"E:\2025 work streams\redo_cafml\output\final_model.pkl")
# with open(model_path, 'rb') as model_file:
#     model_data = joblib.load(model_file)



# from dbfread import DBF
#
# path = r"E:\2025 work streams\caf.brAIn\machine vision\NN input data\04.Final Base Network\NoHAM_Base.DBF"
# table = DBF(path)
# records = [record for record in table]
# df = pd.DataFrame(records)
# print(df)
#
# import geopandas as gpd
#
# path_geo = r"E:\2025 work streams\caf.brAIn\machine vision\NN input data\shapefiles\NoHAM_Base.shp"
# df_geo = gpd.read_file(path_geo)
#
# test = gpd.GeoDataFrame(path_geo, geometry='column name')


# read extra info
# plot image
# make sure correct image corresponds with correct info

# create database of regul


# create geodataframe of each area, find midpoint of each bounding
# have a corresponding satellite image
# based on bng coordinates provided, find that location based on the geodatframes
# then extract that satellite image. this can then be used for two things, one is labelling for training
# second would be finding new junctions that noham wants to verify what they look like



# todo now convert the midpoint to bng so i can now have the midpoint of each satellite image as bng
# so when someone provides coordinates, we find the closest midpoints and then look for that junction?
# or can we just find the exact coordinate based on what we have, will have to check


# todo make each item in the dictionary a list, add the corresponding image to the list as it will have the same name

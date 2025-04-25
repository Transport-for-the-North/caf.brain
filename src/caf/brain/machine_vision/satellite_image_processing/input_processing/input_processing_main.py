# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 4/10/2025
Original author: Adil Zaheer
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path
from caf.brain.machine_vision.satellite_image_processing.image_processing.image_processing_functions import directory_iterator, image_crop
from caf.brain.machine_vision.satellite_image_processing.input_processing.input_processing_functions import process_coordinates


def main_input_processing(geo_df: gpd.geodataframe,
                          df: pd.DataFrame,
                          image_folder: Path,
                          final_html_info: Path,
                          x_coordinate: str,
                          y_coordinate: str,
                          output_path: Path):
    # todo make a flow for generating training images only
    # todo make one for running the entire model? this would be in a main elsewhere?
    # generating training images info just needs this info below?
    coordinate_df, unique_box_boundaries = process_coordinates(geo_df,
                                                               df,
                                                               image_folder,
                                                               final_html_info,
                                                               x_coordinate,
                                                               y_coordinate,
                                                               output_path)

    path_list_of_your_coordinates = directory_iterator(dir_path=image_folder,
                                                       training_data=coordinate_df,
                                                       image_reference_column='box_boundary',
                                                       output=output_path)

    image_crop(path_list=path_list_of_your_coordinates,
               image_supporting_data=coordinate_df,
               output=output_path,
               image_folder=image_folder)

    return



# todo make all this code more efficient overall
# todo, extract coordinates from geometry column
# todo match the coordinates with corresponding british national grid tile
# todo may combine this data with the html stuff? one big data set with corresponding image under the bng tile grid as the dict key?
# todo then i know which junctions (noham base network nodes) are where then can zoom the images
# todo then can start labelling them


# inloop create df that has all the





# junctions
# PM = motorway merge
# R = roundabout
# P = give way
# EP = give way on approach to roundabout
# S = signalised junc
# EX = DROP THIS
# ES = traffic signal on approach to roundabout
# M = mini roundabout
# p = same as P
# s = same as S


# nodes less than 10k are zone connectors (not relevant)


# JUNCTION: ignore None / Z
# ICD (diameter of roundabout), will only have a value if its a roundabout
# drop ucycle, gap, gapm
# MIDLANE = no lanes between one junction and next. so on the way to the junction
# STPLANE = no lanes at stop line of the junction. 1f means two lanes at the junction but one lane is small
# MAJOR = if Y then the road that the major is on is the main road. if signalised, major minor doesnt apply. major minor only applies to give way or give way roundabout

# turns:
# turns work from left to right. so turn one will always be in the left lane
# first number is lane index you can use to turn. second number is how many lanes can turn
#l1-1 means left most lane and only one lane to turn left
# a for ahead. a1-1 means one lane going ahead and you can only use one lane to go ahead

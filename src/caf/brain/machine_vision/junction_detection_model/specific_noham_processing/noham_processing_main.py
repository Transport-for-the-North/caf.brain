# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 3/4/2025
Original author: Adil Zaheer
"""
from caf.brain.machine_vision.junction_detection_model.specific_noham_processing.noham_processing_functions import noham_data


def main_process_noham(output_path,
                       noham_path,
                       geo_path):

    noham_dat = noham_data(geo_path=geo_path,
                           noham_path=noham_path,
                           output=output_path)

    return noham_dat














# data dict key name same as jpg image name DONE
# search data dict for info, that provides coordinates
# set bounds
# zoom to area around coordinates from noham network
# process zoomed image
# output zoomed images so I manually label
# add to data dict with new info to new dict along with corresponding labelled image
# training data finished

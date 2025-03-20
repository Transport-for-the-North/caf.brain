# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 3/3/2025
Original author: Adil Zaheer
"""
from pathlib import Path
import os
import glob
import joblib
import pandas as pd
from tqdm import tqdm
from caf.brain.machine_vision.html_processing.html_processing_functions import extract_info_from_html
import logging
LOG = logging.getLogger(__name__)

def main_process_html(folder_path: Path,
                      output_path: Path) -> dict:
    """
    Creates a dictionary of HTML data based on a path.

    :param folder_path: path to parent folder that contains HTML files.
    :param output_path: path to output location.

    :return: dictionary of data.
    """
    LOG.info('HTML processing beginning')
    data_filename = os.path.join(output_path, 'data_dict.pkl')

    if os.path.exists(data_filename):
        LOG.info('HTML data dictionary already exists and is being read in')
        data_dict = joblib.load(data_filename)
        final_data = pd.read_csv(os.path.join(output_path, "final_data"))
    else:
        data_dict = {}
        counter = 0
        paths = glob.glob(os.path.join(folder_path) + '/**/*.xml', recursive=True)
        with tqdm(total=None) as pbar:
            for path in paths:
                df, name = extract_info_from_html(html_file_path=path)
                data_dict[name] = df
                pbar.update(1)
                counter += 1
                if counter % 1000 == 0:
                    # mod of count value as there is a lot of iterations
                    # 45k for just the north
                    LOG.info(f"Processed item {counter}: {path}")

        joblib.dump(data_dict, data_filename)
        final_data = pd.concat(data_dict.values(), ignore_index=True)
        final_data = pd.DataFrame(final_data)
        # todo change into dataframe then output
        final_data.to_csv(os.path.join(output_path, "final_data"), index=False)
    LOG.info('HTML processing ending')
    return data_dict, final_data



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
from caf.brain.machine_vision.satellite_image_processing.html_processing.html_processing_functions import extract_info_from_html
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
    data_filename = os.path.join(output_path, 'data_dict_html.pkl')
    problem_html = []

    if os.path.exists(data_filename):
        LOG.info('HTML data dictionary already exists and is being read in')
        data_dict = joblib.load(data_filename)
        final_data = pd.read_csv(os.path.join(output_path, "processed_html_data.csv"))
    else:
        data_dict = {}
        counter = 0
        paths = glob.glob(os.path.join(folder_path) + '/**/*.xml', recursive=True)
        with tqdm(total=None) as pbar:
            for path in paths:
                df, name = extract_info_from_html(html_file_path=path)
                if df is None or df.empty:
                    LOG.warning(f"Skipping file {path} due to missing or empty data.")
                    problem_html.append(path)
                    continue
                data_dict[name] = df
                pbar.update(1)
                counter += 1
                if counter % 1000 == 0:
                    # mod of count value as there is a lot of iterations
                    # 45k for just the north
                    LOG.info(f"Processed item {counter}: {path}")

        joblib.dump(data_dict, data_filename)
        final_data = pd.concat(data_dict.values(), ignore_index=True)
        final_data.to_csv(os.path.join(output_path, "processed_html_data.csv"), index=False)

        problem_df = pd.DataFrame(problem_html)
        problem_df.to_csv(os.path.join(output_path, "problem_html_files.csv"), index=False)

    LOG.info('HTML processing ending')
    return data_dict, final_data

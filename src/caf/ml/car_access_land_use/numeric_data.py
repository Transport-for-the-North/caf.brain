# -*- coding: utf-8 -*-
"""
Created on: 1/5/2024
Updated on:

Original author: Adil Zaheer
Last update made by:
Other updates made by:

File purpose:

"""
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import pandas as pd
import numpy as np
import os


def process_data_numeric(data, keep_columns=None, output_path=None):

    # Convert to DataFrame if input is a Python array
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Convert remaining columns to numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Identify non-numeric columns
    non_numeric_columns = data.columns[~data.applymap(np.isreal).all()]

    # Export non-numeric columns to a separate file if output_path is provided
    if output_path:
        output_file_path = os.path.join(output_path, "non_numeric.csv")
        non_numeric_df = data[non_numeric_columns]
        non_numeric_df.to_csv(output_file_path, index=False)
        print(f"Non-numeric columns exported to: {output_file_path}")

    # Set keep_columns to an empty set if not provided
    keep_columns = keep_columns or set()

    # Drop non-numeric columns, but only if they are not in the keep_columns set
    data = data.drop(columns=non_numeric_columns.difference(keep_columns))

    return data


"""
Created on: 9/25/2025
Original author: Adil Zaheer
"""
import os
import pandas as pd
from pathlib import Path
import logging
from caf.brain.machine_vision.temp_folder.generate_user_satellite_images.functions import \
    read_path, euclidean_distance
LOG = logging.getLogger(__name__)


def locate_user_coordinates(user_locations_csv_path: Path,
                            output_path: Path,
                            satellite_metadata: pd.DataFrame,
                            x_coordinate: str | None = None,
                            y_coordinate: str | None = None):
    user_coordinate_filename = "user_coordinate_data.csv"

    if os.path.exists(os.path.join(output_path, user_coordinate_filename)):
        df = pd.read_csv(os.path.join(output_path, user_coordinate_filename))
        return df

    if not user_locations_csv_path:
        raise ValueError("Please provide a csv path with your coordinates. \
                          These can be part of a geography column or two \
                          separate x and y columns")


    user_locations_path = os.path.join(user_locations_csv_path)
    df, file_type = read_path(file_path=user_locations_path)
    if file_type == "shp":
        if 'geometry' not in df.columns:
            raise ValueError("Ensure that geometry column is present in shapefile data.")

        df['coordinates_easting'] = df['geometry'].x
        df['coordinates_northing'] = df['geometry'].y
    else:
        if x_coordinate and y_coordinate not in df.columns:
            raise ValueError("Please provide the column name for your x coordinates \
                              and y coordinates in your data. Alternatively pass \
                              a shapefile with a geometry column")

        df = df.rename(columns={x_coordinate: 'coordinates_easting',
                                y_coordinate: 'coordinates_northing'})

    user_data_with_closest_tile = euclidean_distance(df_a=satellite_metadata,
                                                     df_b=df)






    #todo check if i need user path finder or can just do it with merges
    # todo also need to check how i did the process of separating out images based on classes
    # todo may need to keep that functionality in permanently





    return

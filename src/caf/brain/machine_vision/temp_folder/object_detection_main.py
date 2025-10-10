"""
Created on: 10/10/2025
Original author: Adil Zaheer
"""
from pathlib import Path

from caf.brain.machine_vision.temp_folder.crop_user_satellite_images.main import image_crop
from caf.brain.machine_vision.temp_folder.generate_satellite_image_information.main import \
    image_info_generation
from caf.brain.machine_vision.temp_folder.generate_user_satellite_images.main import \
    locate_user_coordinates


def main(output_path: Path,
         image_folder_path: Path,
         user_locations_csv_path: Path,
         x_coordinate: str | None = None,
         y_coordinate: str | None = None,
         generate_images: bool = False):
    if generate_images:
        satellite_image_metadata = image_info_generation(output_path=output_path,
                                                         image_folder_path=image_folder_path)

        user_image_metadata = locate_user_coordinates(user_locations_csv_path=user_locations_csv_path,
                                                      output_path=output_path,
                                                      satellite_image_metadata=satellite_image_metadata,
                                                      image_folder_path=image_folder_path,
                                                      x_coordinate=x_coordinate,
                                                      y_coordinate=y_coordinate)

        image_crop(user_image_metadata=user_image_metadata,
                   output_path=output_path,
                   satellite_image_metadata=satellite_image_metadata)

    


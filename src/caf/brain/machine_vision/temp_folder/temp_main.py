"""
Created on: 10/10/2025
Original author: Adil Zaheer
"""
import logging

from caf.brain.machine_vision.temp_folder.temp_inputs import object_detection_inputs
from caf.brain.machine_vision.temp_folder.image_creation.crop_user_satellite_images.main import \
    image_crop
from caf.brain.machine_vision.temp_folder.image_creation.generate_satellite_image_information.main import \
    image_info_generation
from caf.brain.machine_vision.temp_folder.image_creation.generate_user_satellite_images.main import \
    locate_user_coordinates

LOG = logging.getLogger(__name__)

# generate images for training
# generate images for using the model


def main(params: object_detection_inputs):
    if params.image_generation_inputs.generate_images:
        satellite_image_metadata = image_info_generation(output_path=params.output_path,
                                                         image_folder_path=params.image_generation_inputs.image_folder_path)

        user_image_metadata = locate_user_coordinates(user_locations_csv_path=params.image_generation_inputs.user_locations_csv_path,
                                                      output_path=params.output_path,
                                                      satellite_image_metadata=satellite_image_metadata,
                                                      image_folder_path=params.image_generation_inputs.image_folder_path,
                                                      x_coordinate=params.image_generation_inputs.x_coordinate,
                                                      y_coordinate=params.image_generation_inputs.y_coordinate)

        _ = image_crop(user_image_metadata=user_image_metadata,
                       output_path=params.output_path,
                       satellite_image_metadata=satellite_image_metadata)

        return

    if params.build_model_inputs.build_model:


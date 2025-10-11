"""
Created on: 7/22/2025
Original author: Adil Zaheer
"""

from caf.brain.machine_vision.object_detection_satellites.satellite_image_processing.image_processing.main_image_crop import (
    image_crop_main,
)
from caf.brain.machine_vision.object_detection_satellites.satellite_image_processing.satellite_metadata_processing.satellite_metadata_main import (
    main_satellite_metadata,
)
import logging

from caf.brain.machine_vision.object_detection_satellites.satellite_image_processing.user_input_processing.process_user_inputs import (
    main_process_user_locations,
)

LOG = logging.getLogger(__name__)


def main_image_generation(
    image_folder_path,
    output_path,
    satellite_image_metadata,
    user_locations_path,
    x_coordinate,
    y_coordinate,
):

    metadata = main_satellite_metadata(
        folder_path=image_folder_path,
        output_path=output_path,
        satellite_image_metadata=satellite_image_metadata,
    )

    user_coordinate_data = main_process_user_locations(
        user_locations_path=user_locations_path,
        output_path=output_path,
        satellite_metadata=metadata,
        image_folder=image_folder_path,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )

    image_crop_main(
        user_coordinate_data=user_coordinate_data,
        output=output_path,
        image_folder=image_folder_path,
        satellite_metadata=metadata,
    )

    return

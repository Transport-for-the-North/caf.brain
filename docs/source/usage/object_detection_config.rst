caf.brAIn Object Detection Configuration Guide
==============================================

This document describes the configuration options for the object detection model (prediction).
A populated config file is always required regardless of whether the tool is run in the
command line or your IDE. This allows you to run the object detection model (either junction detection
or your own model) via the command line.

Configuration File Structure
----------------------------

The configuration file uses YAML format and consists of just one main section:

- ``object_detection``: Paths to model run required files

1. Object detection
-------------------

Defines the locations of model run required directories. All these inputs are
**Required.** to run the model

.. code-block:: yaml

    paths:
        user_location:
        output_path:
        image_folder_path:
        model_path:

Parameters
~~~~~~~~~~

+---------------------+------------------+------------------------------------------------------------------------------------+
| Parameter           | Type             | Description                                                                        |
+=====================+==================+====================================================================================+
| user_location       | str or null      | A path to a shapefile with a geometry column or x and y columns. These are the     |
|                     |                  | locations of where you want to predict. If you are conducting junction detection,  |
|                     |                  | a string of a LSOA, LAD or British town / city can be provided                     |
|                     |                  | (See "B:\TfN_object_detection\Towns_and_Cities" for available cities). All junction|
|                     |                  | locations in that area will be identified.                                         |
+---------------------+------------------+------------------------------------------------------------------------------------+
| output_path         | str or null      | Output directory for all results.                                                  |
+---------------------+------------------+------------------------------------------------------------------------------------+
| image_folder_path   | str or null      | Path to the TfN image folder (B drive connection). Specifically the high           |
|                     |                  | level folder that houses the region folder ("B:\England 12.5cm").                  |
+---------------------+------------------+------------------------------------------------------------------------------------+
| model_path          | str or null      | Path to your trained YOLO model  (best.pt). If you are conducting junction         |
|                     |                  | detection, a trained model exists here "B:\TfN_object_detection\best.pt".          |
+---------------------+------------------+------------------------------------------------------------------------------------+

Notes
~~~~~
- If you are not running junction detection, then you must train the model yourself
  on your specific use case. This can be done via the caf.brAIn object detection
  api which provides two simple functions for generating training images and then
  training a model with those images. Currently the only manual step is labelling
  images for your use case. It is recommended to download and use YoloLabel for this
  process. Caf.brAIn is designed to work with YoloLabel outputs.
  https://github.com/developer0hye/Yolo_Label

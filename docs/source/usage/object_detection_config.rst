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

    object_detection:
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


2. Environment Variables (Optional)
-----------------------------------

In addition to the YAML configuration, the object detection pipeline relies on two
reference datasets:

- OS junction coordinate data
- Towns and Cities boundary data

By default, these datasets are expected at:

- ``B:\TfN_object_detection\os_data.gpkg``
- ``B:\TfN_object_detection\Towns_and_Cities``

These default locations work out-of-the-box for TfN users. However, you may override
these paths using environment variables:

- ``OS_DATA_PATH``
- ``TOWNS_AND_CITIES_PATH``

This allows the tool to run on machines where the B: drive is not available, or where
the datasets are stored in a different location.

Examples
~~~~~~~~

**Windows (PowerShell):**

.. code-block:: powershell

    setx OS_DATA_PATH "C:\data\os_data.gpkg"
    setx TOWNS_AND_CITIES_PATH "C:\data\Towns_and_Cities"

If these variables are not set, the system will fall back to the default B: drive paths.

3. Installation
---------------

This project has specific dependency constraints due to the use of
CUDA-enabled PyTorch, Ultralytics YOLO, Rasterio, and PyProj. As a result,
installation differs slightly depending on whether you use ``pip`` or
``conda``.

Pip Installation (recommended)
------------------------------

Pip supports CUDA-specific PyTorch aspects, so installation works as expected:

.. code-block:: bash

    pip install -r requirements_machine_vision.txt

Notes
~~~~~

- PyTorch and TorchVision are pinned to CUDA 11.8 (``+cu118``). This must match
  your system CUDA driver. If the driver is upgraded, the pinned versions must
  be updated accordingly.
- Ultralytics is pinned due to frequent API changes that affect training and
  evaluation behaviour.
- All other dependencies are compatible with ``>=`` versioning.

Conda Installation
------------------

Conda cannot install the CUDA-specific PyTorch aspects used in this project.
To use conda, create an environment and install all packages via pip:

.. code-block:: bash

    conda create -n cafbrain python=3.10
    conda activate cafbrain
    pip install -r requirements_machine_vision.txt

Notes
~~~~~

- Conda does not support the ``torch==2.3.1+cu118`` or ``torchvision==0.18.1+cu118``
  syntax inside ''requirements_machine_vision.txt''. CUDA-enabled PyTorch must
  be installed via pip.
- TensorFlow Windows builds are not available on ``conda-forge``. If TensorFlow
  is required, it must be installed from Anaconda's default channel or via pip.
- Mixing pip-installed PyTorch with conda-installed PyTorch is not supported.

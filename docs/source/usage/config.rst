caf.brAIn Configuration Guide
=============================

This document describes the configuration options for the full machine learning pipeline.
A populated config file is always required regardless of whether the tool is run in the
command line or your IDE.

Configuration File Structure
----------------------------

The configuration file uses YAML format and consists of four main sections:

- ``paths``: Input/output data locations
- ``data_classification``: Dataset structure and column definitions
- ``transforming_inputs``: Data preprocessing options
- ``modelling``: Model selection and training parameters


1. Paths
--------

Defines the locations of input data and output directories.

.. code-block:: yaml

    paths:
        file_path:
        folder_path:
        output_path:
        validation_path:

Parameters
~~~~~~~~~~

+------------------+------------------+--------------------------------------------------------------+
| Parameter        | Type             | Description                                                  |
+==================+==================+==============================================================+
| file_path        | str or null      | Path to a single CSV or tabular file for model training      |
+------------------+------------------+--------------------------------------------------------------+
| folder_path      | str or null      | Path to folder containing multiple CSVs (same structure)     |
+------------------+------------------+--------------------------------------------------------------+
| output_path      | str or null      | **Required.** Output directory for all results               |
+------------------+------------------+--------------------------------------------------------------+
| validation_path  | str or null      | Path to validation data containing ground truth              |
+------------------+------------------+--------------------------------------------------------------+

Notes
~~~~~

- Either ``file_path`` **or** ``folder_path`` must be provided, not both.
- All CSVs in ``folder_path`` must have identical structures. E.g each CSV represents a different year in your data.
- Pre-processed train/test data should be placed in ``output_path``. This can then be read in and used (e.g. train.csv)
- Arguments that are not populated (null) must be removed from your config file.


2. Data Classification
----------------------

Defines the structure of your dataset and identifies column types.

.. code-block:: yaml

    data_classification:
        target_column: null
        custom_index:
          - null
        categorical_features:
          - null
        numerical_features:
          - null
        weight_column: null
        is_time_series: False

Parameters
~~~~~~~~~~

+------------------------+------------------------+--------------------------------------------------------------+
| Parameter              | Type                   | Description                                                  |
+========================+========================+==============================================================+
| target_column          | str                    | **Required.** Column to predict (dependent variable, Y)      |
+------------------------+------------------------+--------------------------------------------------------------+
| custom_index           | list[str] or null      | Context columns not used for training (e.g. geography, year) |
+------------------------+------------------------+--------------------------------------------------------------+
| categorical_features   | list[str] or null      | Variables with <15 unique values                             |
+------------------------+------------------------+--------------------------------------------------------------+
| numerical_features     | list[str] or null      | Continuous variables with ≥15 unique values                  |
+------------------------+------------------------+--------------------------------------------------------------+
| weight_column          | str or null            | Sample weights for weighted training                         |
+------------------------+------------------------+--------------------------------------------------------------+
| is_time_series         | bool                   | Set ``True`` for temporal data                               |
+------------------------+------------------------+--------------------------------------------------------------+

Notes
~~~~~

- Categorical features are automatically encoded (encoding type can be specified).
- At least one of ``categorical_features`` or ``numerical_features`` must be provided (otherwise the model thinks you have no explanatory data (x)).
- If ``is_time_series`` is ``True``, include time columns in ``custom_index``.
- ``custom_index`` columns help interpretation but do not affect training.
- Remove null arguments from your config file.

3. Transforming Inputs
----------------------

Controls data preprocessing and train/test splitting.

.. code-block:: yaml

    transforming_inputs:
        column_name_to_drop_rows:
          - null
        value_in_row:
          - null
        classification_prediction:
          - null
        split_by_value: null
        split_size: null
        sample_size_encode: False
        select_encode_values: False
        encode_values_to_drop: null
        skip_encoding_and_scaling: False

Parameters
~~~~~~~~~~

+-----------------------------+------------------------------+---------------------------------------------------------------------------+
| Parameter                   | Type                         | Description                                                               |
+=============================+==============================+===========================================================================+
| column_name_to_drop_rows    | list[str] or null            | Columns containing values to filter out                                   |
+-----------------------------+------------------------------+---------------------------------------------------------------------------+
| value_in_row                | list[str/int/float] or null  | Values to remove from corresponding columns (in column_name_to_drop_rows) |
+-----------------------------+------------------------------+---------------------------------------------------------------------------+
| classification_prediction   | list[int] or null            | Classes to predict for classification                                     |
+-----------------------------+------------------------------+---------------------------------------------------------------------------+
| split_by_value              | str or null                  | Value used as train/test split point                                      |
+-----------------------------+------------------------------+---------------------------------------------------------------------------+
| split_size                  | float or null                | Train/test split ratio (default 0.2)                                      |
+-----------------------------+------------------------------+---------------------------------------------------------------------------+
| sample_size_encode          | bool                         | Drop most frequent category during encoding                               |
+-----------------------------+------------------------------+---------------------------------------------------------------------------+
| select_encode_values        | bool                         | Manually specify encoding reference values                                |
+-----------------------------+------------------------------+---------------------------------------------------------------------------+
| encode_values_to_drop       | list[str] or null            | Values to drop when encoding                                              |
+-----------------------------+------------------------------+---------------------------------------------------------------------------+
| skip_encoding_and_scaling   | bool                         | If True, encoding and scaling is skipped                                  |
+-----------------------------+------------------------------+---------------------------------------------------------------------------+

Example: Filtering Rows
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    column_name_to_drop_rows:
      - purpose
      - mode
    value_in_row:
      - 1
      - "car"

This removes rows where ``purpose == 1`` and ``mode == "car"``.

Notes
~~~~~

- Order matters: ``value_in_row`` must match ``column_name_to_drop_rows``.
- Leave encoding options as ``False`` if unsure (standard encoding is applied as a default).
- For time series, use ``split_by_value`` with a date column (unless time ordering does not matter in your modelling).
- ``encode_values_to_drop`` must match the order of ``categorical_features``.
- If choosing a custom encoding option, ensure only one is provided e.g ``sample_size_encode`` to true and removing ``select_encode_values`` and ``encode_values_to_drop`` from the config file.
- Remove null arguments from your config file.

Example: select_encode_values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    select_encode_values: True
    encode_values_to_drop:
      - 0
      - 1
    categorical_features:
      - "mode"
      - "adults_in_household"

4. Modelling
------------

Configures model selection and training parameters.

.. code-block:: yaml

    modelling:
        model_choice:
            - null
        full_transformations: False
        cv: null
        skip_feature_selection: False
        intensive_feature_selection: False

Parameters
~~~~~~~~~~

+-----------------------------+----------------------+--------------------------------------------------------------+
| Parameter                   | Type                 | Description                                                  |
+=============================+======================+==============================================================+
| model_choice                | list[Models]         | **Required.** Models to train                                |
+-----------------------------+----------------------+--------------------------------------------------------------+
| full_transformations        | bool                 | Apply automatic data transformations                         |
+-----------------------------+----------------------+--------------------------------------------------------------+
| cv                          | str or null          | Cross-validation strategy                                    |
+-----------------------------+----------------------+--------------------------------------------------------------+
| skip_feature_selection      | bool                 | Skip feature selection entirely                              |
+-----------------------------+----------------------+--------------------------------------------------------------+
| intensive_feature_selection | bool                 | Apply stricter feature selection                             |
+-----------------------------+----------------------+--------------------------------------------------------------+

Available Models
~~~~~~~~~~~~~~~~

Regression Models:

- ``random_forest_regressor``
- ``extra_trees_regressor``
- ``gradient_boosting_regressor``
- ``adaboost_regressor``
- ``bagging_regressor``
- ``svr``
- ``knn``
- ``ridge``
- ``lasso``
- ``elasticnet``
- ``linear_regressor``
- ``decision_tree_regresor``

Classification Models:

- ``logit_regressor_l1``
- ``logit_regressor_l2``
- ``logit_regressor_elasticnet``
- ``multinomial``
- ``gradient_boosting_classifier``
- ``random_forest_classifier``
- ``extra_trees_classifier``
- ``decision_tree_classifier``
- ``svm_classifier``
- ``xg_boost_classifier``
- ``xg_boost_multi_classifier``

Cross-Validation Options
~~~~~~~~~~~~~~~~~~~~~~~~
- ``stratifiedkfold``
- ``repeatedkfold``
- ``repeatedstratifiedkfold``
- ``timeseriessplit`` (for time series)

Notes
~~~~~
- ``kfold`` is the default cross validation used if no option is provided.
- Do not mix model types (e.g. one from classification and one from regression).
- If ``skip_feature_selection`` and ``intensive_feature_selection`` are false then a standard feature selection is used.
- Feature selection will remove features if they are not predictive. If all feature are required, skipping feature selection is advised.
- Remove null arguments from your config file.
- Both XGBoost variants will not be able to be tested against other algorithms in the same run. If you want to assess their performance against another algorithm, separate runs will have to be done.

Example
~~~~~~~

.. code-block:: yaml

    modelling:
        model_choice:
            - random_forest_regressor
            - gradient_boosting_regressor
        full_transformations: True
        cv: timeseriessplit
        skip_feature_selection: False
        intensive_feature_selection: True


Regression Example Configuration
--------------------------------

.. code-block:: yaml

    paths:
        file_path: "data\\input\\training_data.csv"
        output_path: "outputs\\model_results"
        validation_path: "data\\validation\\validation_2025.csv"

    data_classification:
        target_column: "trip_count"
        custom_index:
          - "year"
          - "geography"
        categorical_features:
          - "household_size"
          - "car_ownership"
        numerical_features:
          - "income"
          - "population_density"
        weight_column: "sample_weight"
        is_time_series: True

    transforming_inputs:
        column_name_to_drop_rows:
          - "purpose"
        value_in_row:
          - 99
        split_by_value: "2020"
        sample_size_encode: False
        select_encode_values: False

    modelling:
        model_choice:
            - random_forest_regressor
            - gradient_boosting_regressor
        full_transformations: True
        cv: timeseriessplit
        skip_feature_selection: False
        intensive_feature_selection: True


Classification Example Configuration
------------------------------------

.. code-block:: yaml

    paths:
        file_path: "data\\input\\training_data.csv"
        output_path: "outputs\\model_results"

    data_classification:
        target_column: 'numcarvan'
        custom_index:
            - "householdid"
            - "soc"
            - "ns"
        categorical_features:
            - "hh_child"
        numerical_features:
            - "trips"
        is_time_series: false

    transforming_inputs:
        classification_prediction:
            - 0
            - 1
        sample_size_encode: false
        select_encode_values: false

    modelling:
        model_choice:
            - logit_regression_elasticnet
            - gradient_boosting_classifier
        full_transformations: true
        skip_feature_selection: false
        intensive_feature_selection: true


Classification Example 2 Configuration
--------------------------------------

.. code-block:: yaml

    paths:
        file_path: "data\\input\\training_data.csv"
        output_path: "outputs\\model_results"

    data_classification:
        target_column: 'numcarvan'
        custom_index:
            - "householdid"
            - "soc"
            - "ns"
        categorical_features:
            - "hh_child"
        numerical_features:
            - "trips"
        is_time_series: true

    transforming_inputs:
        classification_prediction:
            - 0
            - 1
            - 2
            - 3
            - 4
        sample_size_encode: false
        select_encode_values: false
        skip_encoding_and_scaling: true
    modelling:
        model_choice:
            - xg_boost_multi_classifier
        full_transformations: true
        skip_feature_selection: false
        intensive_feature_selection: true


Quick Start Checklist
---------------------

1. Set ``output_path`` (required)
2. Provide either ``file_path`` OR ``folder_path``
3. Define ``target_column``
4. Classify features as ``categorical_features`` or ``numerical_features``
5. Add time columns to ``custom_index`` if ``is_time_series: True`` (geography columns are another example of what should be in ``custom_index``)
6. Choose at least one model in ``model_choice``
7. Use ``cv: timeseriessplit`` for time series data
8. Remove null arguments from your config file


Common Pitfalls
---------------

1. Forgetting ``is_time_series: True`` for temporal data
2. Not including time columns in ``custom_index`` for time series
3. Mixing regression and classification models
4. Leaving ``target_column`` as null
5. Mismatched lengths between ``column_name_to_drop_rows`` and ``value_in_row``
6. Leaving null values in the config file

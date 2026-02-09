# caf.brAIn Configuration Guide

This document describes the configuration options for the full machine learning pipeline.  
A populated config file is always required regardless of if the tool is run in the command line or your IDE.

## Configuration File Structure

The configuration file uses YAML format and consists of four main sections:
- `paths`: Input/output data locations
- `data_classification`: Dataset structure and column definitions
- `transforming_inputs`: Data preprocessing options
- `modelling`: Model selection and training parameters

---

## 1. Paths

Defines the locations of input data and output directories.

```yaml
paths:
    file_path: 
    folder_path: 
    output_path: 
    validation_path: 
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` or `null` | Path to a single CSV or tabular file containing input data for model training |
| `folder_path` | `str` or `null` | Path to a folder containing multiple CSV files (all must have the same structure). Alternative to `file_path` |
| `output_path` | `str` or `null` | **Required.** Path where all outputs will be saved. If using pre-split train/test data, place them here |
| `validation_path` | `str` or `null` | Path to validation data file containing ground truth values for evaluation |

### Notes
- Either `file_path` OR `folder_path` must be provided, not both
- All CSVs in `folder_path` must have identical column structures
- For consistency, place pre-processed train/test data in `output_path`
- Arguments that are not populated (null) must be removed from your config file.
---

## 2. Data Classification

Defines the structure of your dataset and identifies column types.

```yaml
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
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_column` | `str` | **Required.** The column you want to predict (dependent variable, Y) |
| `custom_index` | `list[str]` or `null` | Columns that provide context but aren't used for training (e.g., year, geography) |
| `categorical_features` | `list[str]` or `null` | Non-continuous variables with <15 unique values (e.g., household adults: 0, 1, 2) |
| `numerical_features` | `list[str]` or `null` | Continuous variables with ≥15 unique values (e.g., income, temperature, trips) |
| `weight_column` | `str` or `null` | Column containing sample weights for weighted model training |
| `is_time_series` | `bool` | Set to `True` if data has a temporal component, `False` otherwise |

### Notes
- **Categorical features** will be automatically encoded during preprocessing
- **Custom index** columns should include any time columns (year, month, day etc.) if `is_time_series` is `True`
- Include custom index columns to aid interpretation without affecting model training
- Arguments that are not populated (null) must be removed from your config file.
---

## 3. Transforming Inputs

Controls data preprocessing and train/test splitting.

```yaml
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
```

### Parameters

| Parameter | Type                            | Description                                                                                                       |
|-----------|---------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `column_name_to_drop_rows` | `list[str]` or `null`           | Columns containing values to be filtered out                                                                      |
| `value_in_row` | `list[str/float/int]` or `null` | Values to remove from corresponding columns in `column_name_to_drop_rows`                                         |
| `classification_prediction` | `list[int]` or `null`           | Classes to predict for classification problems (e.g., `0, 1, 2`)                                                  |
| `split_by_value` | `str` or `null`                 | Value in a custom index column to use as train/test split point (e.g., "2020" for year)                           |
| `split_size` | `float` or `null`               | Train/test split ratio (default: 0.2 if null)                                                                     |
| `sample_size_encode` | `bool`                          | If `True`, encode categorical variables by dropping the most frequent category                                    |
| `select_encode_values` | `bool`                          | If `True`, manually specify which categorical values to drop during encoding                                      |
| `encode_values_to_drop` | `list[str]` or `null`           | Categorical values to drop when encoding (length must match `categorical_features`). Links to select_encode_values. |

### Example: Filtering Rows

```yaml
column_name_to_drop_rows:
  - purpose
  - mode
value_in_row:
  - 1
  - "car"
```
This removes all rows in the purpose column where `purpose == 1` and   
in the mode column where `mode == "car"`.

### Notes
- Order matters: `value_in_row` entries must correspond to `column_name_to_drop_rows` entries
- If unfamiliar with encoding, leave `sample_size_encode` and `select_encode_values` as `False`
- For time series data, use `split_by_value` with a year/date column in `custom_index`
- If `sample_size_encode`, `select_encode_values` and `encode_values_to_drop` are left as False and null respectively, standard SciKitLearn encoding principles are applied.
- If `select_encode_values` is true, then `encode_values_to_drop` must contain the values to "drop" (used as reference for encoding). This must also be in the same order as the variables passed to `categorical_features`. Example below.
- Arguments that are not populated (null) must be removed from your config file.

### Example: select_encode_values
```yaml
select_encode_values: True
encode_values_to_drop:
  - 0
  - 1
categorical_features: 
  - "mode"
  - "adults_in_household"
```
- So here mode 0 and adults_in_household 1 are dropped (used as reference values) during encoding. This allows you to have more control over categorical variable encoding.
---

## 4. Modelling

Configures model selection and training parameters.

```yaml
modelling:
    model_choice:
        - null
    full_transformations: False
    cv: null
    skip_feature_selection: False
    intensive_feature_selection: False
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_choice` | `list[Models]` | **Required.** List of models to train (see available models below) |
| `full_transformations` | `bool` | If `True`, apply automatic transformations to fix data issues |
| `cv` | `str` or `null` | Cross-validation strategy (default: standard K-Fold if null) |
| `skip_feature_selection` | `bool` | If `True`, skip feature selection entirely |
| `intensive_feature_selection` | `bool` | If `True`, apply stricter feature selection (falls back to standard if needed) |

### Available Models

Specify models using the `Models` enum format (e.g., `Models.RANDOM_FOREST_REGRESSOR`):

**Regression Models:**
- `random_forest_regressor`
- `extra_trees_regressor`
- `gradient_boosting_regressor`
- `adaboost_regressor`
- `bagging_regressor`
- `svr`
- `knn`
- `ridge`
- `lasso`
- `elasticnet`
- `linear_regressor`
- `decision_tree_regresor`

**Classification Models:**
- `logit_regressor_l1`
- `logit_regressor_l2`
- `logit_regressor_elasticnet`
- `multinomial`
- `gradient_boosting_classifier`
- `random_forest_classifier`
- `extra_trees_classifier`
- `decision_tree_classifier`
- `svm_classifier`

### Cross-Validation Options

If `cv` is `null`, standard K-Fold is used. Available options:
- `stratifiedkfold` - Maintains class distribution in each fold
- `repeatedkfold` - Repeats K-Fold multiple times
- `repeatedstratifiedkfold` - Combines repeated and stratified
- `timeseriessplit` - **Use for time series data**

### Example

```yaml
modelling:
    model_choice:
        - random_forest_regressor
        - gradient_boosting_regressor
    full_transformations: True
    cv: timeseriessplit
    skip_feature_selection: False
    intensive_feature_selection: True
```

### Notes
- You are able to provide a list of one algorithm to `model_choice` if you know which you would like to use
- Classification algorithms are used for categorical Y variables (e.g. number of cars in a household). Regression is used for continuous Y variables (e.g. number of trips taken).
- Leaving `skip_feature_selection` and `intensive_feature_selection` as False still allows standard feature selection to run. 
- Arguments that are not populated (null) must be removed from your config file.

---

## Regression Example Configuration

```yaml
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
```
---
## Classification Example Configuration

```yaml
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
```

---
## Quick Start Checklist

1. Set `output_path` (required)
2. Provide either `file_path` OR `folder_path`
3. Define `target_column`
4. Classify features as `categorical_features` or `numerical_features`
5. Add time columns to `custom_index` if `is_time_series: True`
6. Choose at least one model in `model_choice`
7. Set `cv: timeseriessplit` if working with time series data
8. Arguments that are not populated (null) are removed from your config file.
---

## Common Pitfalls

1. Forgetting to set `is_time_series: True` when working with temporal data  
2. Not including time columns in `custom_index` for time series problems  
3. Mixing regression and classification models - choose models appropriate for your problem  
4. Leaving `target_column` as null - this is required  
5. Mismatched lengths between `column_name_to_drop_rows` and `value_in_row`
6. Leaving values as null / None and not removing them from the config
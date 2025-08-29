"""
Main function to tidy input data.
"""

# Built-Ins
import logging
import pathlib

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml.functions_and_classes.process_data_functions.encode_and_scale import (
    process_data_pipeline,
)
from caf.brain.ml.functions_and_classes.process_data_functions.input_data import (
    InitialDataProcessing,
)
from caf.brain.ml.functions_and_classes.process_data_functions.split_data_into_ttv import (
    split_data,
)
from caf.brain.ml.inputs_and_baseclasses.ml_inputs import PredictionModelInputs

LOG = logging.getLogger(__name__)


def main_input_data(
    paths: PredictionModelInputs.Paths,
    data_classification: PredictionModelInputs.DataClassificationInputs,
    transforming_inputs: PredictionModelInputs.TransformingInputDataInputs,
    output_path,
) -> dict:
    """
    Main function for processing input data for machine learning modelling.

    Parameters
    ----------
    paths: Path inputs from the PredictionModelInputs class. These inputs
           define paths to external files. See
           caf/brain/ml/main_models/prediction_model/ml_inputs.py
           for available options.
    data_classification: Data classification inputs from the PredictionModelInputs
                         class. These inputs help define and outline the
                         structure of the input data. See
                         caf/brain/ml/main_models/prediction_model/ml_inputs.py
                         for available options.
    transforming_inputs: Transforming inputs from the PredictionModelInputs
                         class. These inputs dictate how the data is transformed
                         for machine learning modelling. See
                         caf/brain/ml/main_models/prediction_model/ml_inputs.py
                         for available options.
    output_path: Path to output location.

    Returns
    -------
    data_dict: Dictionary of processed dataframes.
    drop_vals: Values dropped during encoding of categorical variables.
    numerical_pipeline: SciKit-Learn pipeline that contains scaling information.
    """

    folder_path = pathlib.Path("" if paths.folder_path is None else paths.folder_path)
    output_path = pathlib.Path(output_path)

    if (output_path / "train.csv").exists() or (folder_path / "train.csv").exists():
        try:
            train_raw = pd.read_csv(output_path / "train.csv")
            test_raw = pd.read_csv(output_path / "test.csv")
            try:
                validate = pd.read_csv(output_path / "validate.csv")
                validate[data_classification.target_column] = validate[
                    data_classification.target_column
                ].astype(float)
            except FileNotFoundError:
                LOG.warning("Validate not provided. Validation will not be performed")
                validate = None

        except FileNotFoundError:
            train_raw = pd.read_csv(folder_path / "train.csv")
            test_raw = pd.read_csv(folder_path / "test.csv")
            try:
                validate = pd.read_csv(folder_path / "validate.csv")
                validate[data_classification.target_column] = validate[
                    data_classification.target_column
                ].astype(float)
            except FileNotFoundError:
                LOG.warning("Validate not provided. Validation will not be performed")
                validate = None

        processor = InitialDataProcessing(
            file_path=paths.file_path,
            folder_path=folder_path,
            output_path=output_path,
            target_column=data_classification.target_column,
            custom_index=data_classification.custom_index,
            column_name_to_drop_rows=transforming_inputs.column_name_to_drop_rows,
            value_in_row=transforming_inputs.value_in_row,
            weight_column=data_classification.weight_column,
            categorical_features=data_classification.categorical_features,
            numerical_features=data_classification.numerical_features,
            classification_prediction=transforming_inputs.classification_prediction,
        )

        processed_dfs = {}
        for name, df in [("train", train_raw), ("test", test_raw)]:
            processor.df = df.copy()
            processor.dataframes = {}

            current_name = name
            is_test_data = current_name.lower() == "test"

            processed = processor.data_already_split_pipeline(is_test_data)

            processed_df = list(processed.values())[0]
            processed_dfs[name] = processed_df

            LOG.info("Processed %s dataframe:", name)
            LOG.info("Index names: %s", processed_df.index.names)
            LOG.info("Columns: %s", processed_df.columns.tolist())
            LOG.info("Shape: %s", processed_df.shape)

        train_unscaled = processed_dfs["train"]
        test_unscaled = processed_dfs["test"]
        train_unscaled[data_classification.target_column] = train_unscaled[
            data_classification.target_column
        ].astype(int)

        train_scaled, drop_vals, numerical_pipeline = process_data_pipeline(
            df=train_unscaled.copy(),
            numerical_features=data_classification.numerical_features,
            categorical_features=data_classification.categorical_features,
            target_column=data_classification.target_column,
            weight_column=data_classification.weight_column,
            sample_size_encode=transforming_inputs.sample_size_encode,
            select_encode_values=transforming_inputs.select_encode_values,
            encode_values_to_drop=transforming_inputs.encode_values_to_drop,
            train_encoded=None,
            test_data=False,
            numerical_pipeline=None,
            output_folder=output_path,
        )

        test_scaled, _, _ = process_data_pipeline(
            df=test_unscaled.copy(),
            numerical_features=data_classification.numerical_features,
            categorical_features=data_classification.categorical_features,
            target_column=data_classification.target_column,
            weight_column=data_classification.weight_column,
            sample_size_encode=transforming_inputs.sample_size_encode,
            select_encode_values=transforming_inputs.select_encode_values,
            encode_values_to_drop=transforming_inputs.encode_values_to_drop,
            train_encoded=train_scaled,
            test_data=True,
            numerical_pipeline=numerical_pipeline,
            output_folder=output_path,
        )

        data_dict = {
            "train_scaled": train_scaled,
            "test_scaled": test_scaled,
            "train_unscaled": train_unscaled,
            "test_unscaled": test_unscaled,
            "validate": validate,
        }
        return data_dict, drop_vals, numerical_pipeline

    processor = InitialDataProcessing(
        file_path=paths.file_path,
        folder_path=folder_path,
        output_path=output_path,
        target_column=data_classification.target_column,
        custom_index=data_classification.custom_index,
        column_name_to_drop_rows=transforming_inputs.column_name_to_drop_rows,
        value_in_row=transforming_inputs.value_in_row,
        weight_column=data_classification.weight_column,
        categorical_features=data_classification.categorical_features,
        numerical_features=data_classification.numerical_features,
        classification_prediction=transforming_inputs.classification_prediction,
    )

    is_test_data = False
    processed_dataframes = processor.execute_pipeline(is_test_data)
    if len(processed_dataframes) == 1:
        df = list(processed_dataframes.values())[0]
    else:
        df = pd.concat(processed_dataframes.values(), axis=0)

    train_unscaled, test_unscaled, validate = split_data(
        processed_dataframes=df,
        paths=paths,
        data_classification=data_classification,
        transforming_inputs=transforming_inputs,
        output_path=output_path,
    )

    if validate is not None:
        validate[data_classification.target_column] = validate[
            data_classification.target_column
        ].astype(int)

    train_scaled, drop_vals, numerical_pipeline = process_data_pipeline(
        df=train_unscaled.copy(),
        numerical_features=data_classification.numerical_features,
        categorical_features=data_classification.categorical_features,
        target_column=data_classification.target_column,
        weight_column=data_classification.weight_column,
        sample_size_encode=transforming_inputs.sample_size_encode,
        select_encode_values=transforming_inputs.select_encode_values,
        encode_values_to_drop=transforming_inputs.encode_values_to_drop,
        train_encoded=None,
        test_data=False,
        numerical_pipeline=None,
        output_folder=output_path,
    )

    test_scaled, _, _ = process_data_pipeline(
        df=test_unscaled.copy(),
        numerical_features=data_classification.numerical_features,
        categorical_features=data_classification.categorical_features,
        target_column=data_classification.target_column,
        weight_column=data_classification.weight_column,
        sample_size_encode=transforming_inputs.sample_size_encode,
        select_encode_values=transforming_inputs.select_encode_values,
        encode_values_to_drop=transforming_inputs.encode_values_to_drop,
        train_encoded=train_scaled,
        test_data=True,
        numerical_pipeline=numerical_pipeline,
        output_folder=output_path,
    )

    train_unscaled[data_classification.target_column] = train_unscaled[
        data_classification.target_column
    ].astype(int)
    train_scaled[data_classification.target_column] = train_scaled[
        data_classification.target_column
    ].astype(int)

    data_dict = {
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "train_unscaled": train_unscaled,
        "test_unscaled": test_unscaled,
        "validate": validate,
    }

    return data_dict, drop_vals, numerical_pipeline

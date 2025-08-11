# Built-Ins
import logging
import os
from pathlib import Path

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml.functions_and_classes.data_analysis.functions import (
    pre_forecast_data_analysis,
)
from caf.brain.ml.functions_and_classes.model_selection.functions import (
    initialise_model,
)
from caf.brain.ml.functions_and_classes.process_data_functions.main import (
    main_input_data,
)
from caf.brain.ml.functions_and_classes.process_data_functions.split_data_into_ttv import (
    simple_train_test_split,
)
from caf.brain.ml.inputs_and_baseclasses.ml_inputs import PredictionModelInputs

LOG = logging.getLogger(__name__)


def main_evaluate_input_data(
    paths: PredictionModelInputs.Paths,
    data_classification: PredictionModelInputs.DataClassificationInputs,
    transforming_inputs: PredictionModelInputs.TransformingInputDataInputs,
    modelling: PredictionModelInputs.ModellingInputs,
    output_path: Path = None,
    train_scaled: pd.DataFrame = None,
    test_scaled: pd.DataFrame = None,
    train_unscaled: pd.DataFrame = None,
    test_unscaled: pd.DataFrame = None,
    model_fit: object = None,
    model_initialised: object = None,
    residuals: pd.Series = None,
    x_train: pd.DataFrame = None,
    x_test: pd.DataFrame = None,
    numerical_pipeline: object = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function for testing data quality with traditional statistics

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
    modelling: Modelling inputs from the PredictionModelInputs
               class. These inputs control the machine learning modelling
               pipeline and functions. See
               caf/brain/ml/main_models/prediction_model/ml_inputs.py
               for available options.
    output_path: Path to output location.
    train_scaled: Processed input data split into train set.
    test_scaled: Processed input data split into train set.
    train_unscaled: Input data unprocessed split into train.
    test_unscaled: Input data unprocessed split into test.
    model_fit: Fitted model on train_test_split test data.
    model_initialised: Initialised SciKitLearn model.
    residuals: Truth values form the train_test_split against the predictions.
    x_train: Series of train data to be used as train.
    x_test: Series of test data to be used as unseen test data.
    numerical_pipeline: Stored numerical transformation pipeline for
                        full model runs. Left as None if not a full
                        model run.
    Returns
    -------
    train_transformed: training data with the numerical features transformed
                       or train_scaled if transformations not applied.
    test_transformed: test data with the numerical features transformed
                      or test_scaled if transformations not applied.
    """
    if train_scaled is None and train_unscaled is None:
        output_path = paths.output_path / "output"
        if not output_path.exists():
            os.makedirs(output_path)

    if train_scaled is None and train_unscaled is None:
        LOG.info("Evaluation of input data beginning. Outputs are saved to: %s", output_path)

        data_dict, _, numerical_pipeline = main_input_data(
            paths, data_classification, transforming_inputs, output_path=output_path
        )

        train_scaled = pd.DataFrame.from_dict(data_dict["train_scaled"])
        test_scaled = pd.DataFrame.from_dict(data_dict["test_scaled"])
        train_unscaled = pd.DataFrame.from_dict(data_dict["train_unscaled"])
        test_unscaled = pd.DataFrame.from_dict(data_dict["test_unscaled"])
        x_train, x_test, y_train, y_test, x_train_weight = simple_train_test_split(
            df=train_unscaled,
            target_column=data_classification.target_column,
            weight_column=data_classification.weight_column,
        )

        if len(modelling.model_choice) > 1:
            LOG.error(
                "You have passed more than one algorithm to the function. \
                       Only provide one algorithm when using this function standalone. \
                       Please use either the full prediction model pipeline or \
                       use the model_selection_main function"
            )
            raise ValueError("Multiple algorithms provided")

        model_initialised = modelling.model_choice[0].get_model()

        model_fit, residuals, _ = initialise_model(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            x_train_weight=x_train_weight,
            output_folder=output_path,
            model_initialised=model_initialised,
            classification_prediction=transforming_inputs.classification_prediction,
        )

        train_transformed, test_transformed = pre_forecast_data_analysis(
            data_classification=data_classification,
            modelling=modelling,
            output_folder=output_path,
            residuals=residuals,
            model_fit=model_fit,
            model_initialised=model_initialised,
            x_train=x_train,
            x_test=x_test,
            train_scaled=train_scaled,
            test_scaled=test_scaled,
            train_unscaled=train_unscaled,
            test_unscaled=test_unscaled,
            numerical_pipeline=numerical_pipeline,
        )
        train_transformed.to_csv(output_path / "train_transformed.csv")
        test_transformed.to_csv(output_path / "test_transformed.csv")

        LOG.info("Evaluation of input data finished.")
        return train_transformed, test_transformed

    train_transformed, test_transformed = pre_forecast_data_analysis(
        data_classification=data_classification,
        modelling=modelling,
        output_folder=output_path,
        residuals=residuals,
        model_fit=model_fit,
        model_initialised=model_initialised,
        x_train=x_train,
        x_test=x_test,
        train_scaled=train_scaled,
        test_scaled=test_scaled,
        train_unscaled=train_unscaled,
        test_unscaled=test_unscaled,
        numerical_pipeline=numerical_pipeline,
    )

    if train_transformed is not None and test_transformed is not None:
        return train_transformed, test_transformed

    return train_scaled, test_scaled

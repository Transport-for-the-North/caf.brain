"""
Main function used to apply data analysis and corrective measures where
applicable.
"""

# Built-Ins
import logging
import os
from pathlib import Path

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml._functions._ml_inputs import (
    DataClassificationInputs,
    ModellingInputs,
    Models,
    Paths,
    TransformingInputDataInputs,
)
from caf.brain.ml._functions.data_analysis.functions import (
    pre_forecast_data_analysis,
)
from caf.brain.ml._functions.model_selection.functions import (
    initialise_model,
)
from caf.brain.ml._functions.process_data_functions.main import (
    main_input_data,
)
from caf.brain.ml._functions.process_data_functions.split_data_into_ttv import (
    simple_train_test_split,
)

LOG = logging.getLogger(__name__)


def main_evaluate_input_data(
    paths: Paths,
    data_classification: DataClassificationInputs,
    transforming_inputs: TransformingInputDataInputs,
    modelling: ModellingInputs,
    output_path: Path | None = None,
    train_scaled: pd.DataFrame | None = None,
    test_scaled: pd.DataFrame | None = None,
    train_unscaled: pd.DataFrame | None = None,
    test_unscaled: pd.DataFrame | None = None,
    model_fit: object = None,
    model_initialised: object = None,
    residuals: pd.Series | None = None,
    x_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    numerical_pipeline: object = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function for testing data quality with traditional statistics

    Parameters
    ----------
    paths: Path inputs from the PredictionModelInputs class. These inputs
           define paths to external files. See
           caf/brain/ml/main_models/prediction_model/_ml_inputs.py
           for available options.
    data_classification: Data classification inputs from the PredictionModelInputs
                         class. These inputs help define and outline the
                         structure of the input data. See
                         caf/brain/ml/main_models/prediction_model/_ml_inputs.py
                         for available options.
    transforming_inputs: Transforming inputs from the PredictionModelInputs
                         class. These inputs dictate how the data is transformed
                         for machine learning modelling. See
                         caf/brain/ml/main_models/prediction_model/_ml_inputs.py
                         for available options.
    modelling: Modelling inputs from the PredictionModelInputs
               class. These inputs control the machine learning modelling
               pipeline and _functions. See
               caf/brain/ml/main_models/prediction_model/_ml_inputs.py
               for available options.
    output_path: Path to output location.
    train_scaled: Processed input data split into train set.
    test_scaled: Processed input data split into train set.
    train_unscaled: Input data unprocessed split into train.
    test_unscaled: Input data unprocessed split into test.
    model_fit: Fitted model on train_test_split test data.
    model_initialised: Initialised SciKitLearn model.
    residuals: Truth values from the train_test_split against the predictions.
    x_test: Dataframe of test data to be used as unseen test data.
    y_test: Series of target data from train, test split.
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
    if not output_path:
        if paths.output_path is None:
            raise ValueError(
                "paths.output_path must not be None when output_path is not provided."
            )
        output_path = paths.output_path / "output"
        if not output_path.exists():
            os.makedirs(output_path)

    if train_scaled is None and train_unscaled is None:
        LOG.info("Evaluation of input data beginning. Outputs are saved to: %s", output_path)

        data_dict, _, numerical_pipeline = main_input_data(
            paths,
            data_classification,
            transforming_inputs,
            output_path=output_path,
            modelling=modelling,
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

        if not modelling.model_choice or len(modelling.model_choice) > 1:
            raise ValueError(
                "Please provide only one algorithm to be used to \
                              evaluate input data if using the function standalone."
            )
        # model re-initialised as this flow is assuming you aren't running the full model
        model_initialised = modelling.model_choice[0].get_model()

        if modelling.model_choice[0] == Models.XGBOOST_MULTICLASS:
            num_classes = train_scaled[data_classification.target_column].nunique()
            model_initialised.set_params(num_class=num_classes)

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
            x_test=x_test,
            y_test=y_test,
            train_scaled=train_scaled,
            test_scaled=test_scaled,
            train_unscaled=train_unscaled,
            test_unscaled=test_unscaled,
            numerical_pipeline=numerical_pipeline,
        )
        train_transformed.to_csv(
            output_path / "final_train.csv",
            index=not isinstance(train_transformed.index, pd.RangeIndex)
        )
        test_transformed.to_csv(
            output_path / "final_test.csv",
            index=not isinstance(test_transformed.index, pd.RangeIndex)
        )

        LOG.info("Evaluation of input data finished.")
        return train_transformed, test_transformed

    if (
        train_scaled is None
        or test_scaled is None
        or train_unscaled is None
        or test_unscaled is None
    ):
        raise ValueError(
            "An error has occurred with this function. Train and test scaled \n"
            "and unscaled are none but should have either been provided directly \n"
            "or generated by the paths (config) to your data.\n"
            "Please re run the function and check your provided paths and arguments."
        )

    train_transformed, test_transformed = pre_forecast_data_analysis(
        data_classification=data_classification,
        modelling=modelling,
        output_folder=output_path,
        residuals=residuals,
        model_fit=model_fit,
        model_initialised=model_initialised,
        x_test=x_test,
        y_test=y_test,
        train_scaled=train_scaled,
        test_scaled=test_scaled,
        train_unscaled=train_unscaled,
        test_unscaled=test_unscaled,
        numerical_pipeline=numerical_pipeline,
    )

    train_transformed.to_csv(
        output_path / "final_train.csv",
        index=not isinstance(train_transformed.index, pd.RangeIndex)
    )
    test_transformed.to_csv(
        output_path / "final_test.csv",
        index=not isinstance(test_transformed.index, pd.RangeIndex)
    )

    if train_transformed is not None and test_transformed is not None:
        return train_transformed, test_transformed

    return train_scaled, test_scaled

"""
Main function for model selection capabilities.
"""

# Built-Ins
import logging
from pathlib import Path

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml.functions.model_selection.functions import select_model
from caf.brain.ml.functions.process_data_functions.main import (
    main_input_data,
)
from caf.brain.ml.inputs_and_baseclasses.ml_inputs import PredictionModelInputs

LOG = logging.getLogger(__name__)


def main_model_selection(
    data_classification: PredictionModelInputs.DataClassificationInputs,
    transforming_inputs: PredictionModelInputs.TransformingInputDataInputs,
    modelling: PredictionModelInputs.ModellingInputs,
    output: Path,
    paths: PredictionModelInputs.Paths,
    train: pd.DataFrame = None,
) -> object:
    """
    Function to automatically score and rank algorithms from the Models
    class which can be used in machine learning modelling.

    Parameters
    ----------
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
    paths: Path inputs from the PredictionModelInputs class. These inputs
           define paths to external files. See
           caf/brain/ml/main_models/prediction_model/ml_inputs.py
           for available options.
    output: Path to output location.
    train: Processed input data split into train subset.

    Returns
    -------
    Selected_model: The best performing SciKitLearn model based on the user
                    provided list. This model is not initialised and ready
                    for further use.
    """
    if train is None:
        LOG.info(
            "Train is none so data is being read in from PredictionModelInputs.Paths.file_path"
        )

        data_dict, _, _ = main_input_data(
            output_path=output,
            paths=paths,
            data_classification=data_classification,
            transforming_inputs=transforming_inputs,
        )
        LOG.info("Data successfully read in, processed and validated")
        train = pd.DataFrame.from_dict(data_dict["train_scaled"])
    if not modelling.model_choice:
        raise ValueError(
            "Please provide at least one model from the Models \
                          class."
        )

    if not isinstance(modelling.model_choice, list):
        model = [modelling.model_choice]
    else:
        model = modelling.model_choice

    if len(model) == 1:
        selected_model = model[0].get_model()

    elif len(model) > 1:
        LOG.info("Beginning model evaluation.")
        selected_model = select_model(
            train=train,
            target_column=data_classification.target_column,
            weight_column=data_classification.weight_column,
            models_to_test=model,
            classification_prediction=transforming_inputs.classification_prediction,
            output_folder=output,
        )
    else:
        LOG.error(
            "Model incorrectly provided or not provided at all \
                          Provide a valid model(s) from the Models Enum class."
        )
        raise ValueError(
            "Model incorrectly provided or not provided at all \
                          Provide a valid model(s) from the Models Enum class."
        )

    return selected_model

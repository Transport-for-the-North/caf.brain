"""
Created on: 1/21/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
from pathlib import Path

# Third Party
import pandas as pd

# Local Imports
from caf.brain.ml._functions._ml_inputs import (
    DataClassificationInputs,
    ModellingInputs,
    Paths,
    TransformingInputDataInputs,
)
from caf.brain.ml._functions.hparam_optimisation.functions import (
    select_param,
)
from caf.brain.ml._functions.model_selection.main import main_model_selection
from caf.brain.ml._functions.process_data_functions.main import (
    main_input_data,
)

LOG = logging.getLogger(__name__)


def main_hyperparameter_optimisation(
    paths: Paths,
    data_classification: DataClassificationInputs,
    transforming_inputs: TransformingInputDataInputs,
    modelling: ModellingInputs,
    output_folder: Path,
    train: pd.DataFrame = None,
    model_instance=None,
):
    """
    Main function for hyperparameter optimisation.

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
    train: Dataframe of final training data post feature selection.
    model_instance: Initialised model algorithm from Models enum class.
    output_folder: Path to output location.

    Returns
    -------
    best_model: Fitted final model for prediction on unseen (test) data.
    """
    if train is None:
        LOG.info("Train does not exist so is being generated with the main_input_data \
                  function based on user provided inputs")
        data_dict, _, _ = main_input_data(
            output_path=output_folder,
            paths=paths,
            data_classification=data_classification,
            transforming_inputs=transforming_inputs,
        )

        train = pd.DataFrame.from_dict(data_dict["train_scaled"])

    if model_instance is None:
        LOG.info("Model instance is None so main_model_selection is being called to \
                  obtain the initialised model")
        model_instance = main_model_selection(
            paths=paths,
            data_classification=data_classification,
            transforming_inputs=transforming_inputs,
            modelling=modelling,
            train=train,
            output=output_folder,
        )

    LOG.info("Hyperparameter optimisation underway")
    best_model = select_param(
        train_final=train,
        target_column=data_classification.target_column,
        model_instance=model_instance,
        model_name=modelling.model_choice,
        classification_prediction=transforming_inputs.classification_prediction,
        cv=modelling.cv,
        weight_column=data_classification.weight_column,
        output_folder=output_folder,
        is_time_series=data_classification.is_time_series,
    )
    return best_model

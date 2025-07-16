"""
Created on: 1/21/2025
Original author: Adil Zaheer
"""
from caf.brain.ml.hyperparameter_optimisation.hyper_optim_functions import select_param
from pathlib import Path
import pandas as pd

from caf.brain.ml.main_models.prediction_model.prediction_model_inputs import PredictionModelInputs


def main_hyperparameter_optimisation(
    paths: PredictionModelInputs.Paths,
    data_classification: PredictionModelInputs.DataClassificationInputs,
    transforming_inputs: PredictionModelInputs.TransformingInputDataInputs,
    modelling: PredictionModelInputs.ModellingInputs,
    train_final: pd.DataFrame = None,
    model_instance=None,
    output_folder: Path = None,
):
    """
    Main function for hyperparameter optimisation.

    Parameters
    ----------
    paths: Path inputs from the PredictionModelInputs class. These inputs
           define paths to external files. See
           caf/brain/ml/main_models/prediction_model/prediction_model_inputs.py
           for available options.
    data_classification: Data classification inputs from the PredictionModelInputs
                         class. These inputs help define and outline the
                         structure of the input data. See
                         caf/brain/ml/main_models/prediction_model/prediction_model_inputs.py
                         for available options.
    transforming_inputs: Transforming inputs from the PredictionModelInputs
                         class. These inputs dictate how the data is transformed
                         for machine learning modelling. See
                         caf/brain/ml/main_models/prediction_model/prediction_model_inputs.py
                         for available options.
    modelling: Modelling inputs from the PredictionModelInputs
               class. These inputs control the machine learning modelling
               pipeline and functions. See
               caf/brain/ml/main_models/prediction_model/prediction_model_inputs.py
               for available options.
    train_final: Dataframe of final training data post feature selection.
    model_instance: Initialised model algorithm from Models enum class.
    output_folder: Path to output location.

    Returns
    -------
    best_model: Fitted final model for prediction on unseen (test) data.
    """

    best_model = select_param(
        train_final=train_final,
        target_column=data_classification.target_column,
        model_instance=model_instance,
        model_name=modelling.model_choice,
        classification_prediction=transforming_inputs.classification_prediction,
        cv=modelling.cv,
        weight_column=data_classification.weight_column,
        output_folder=output_folder,
        is_time_series=data_classification.is_time_series,
    )
    # todo return not just the best model but also the best grid as a dictionary for people to use?
    return best_model

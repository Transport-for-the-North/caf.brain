"""
Main function for model selection capabilities.
"""

# Built-Ins
import logging
from pathlib import Path

# Third Party
import pandas as pd
from sklearn.base import BaseEstimator

# Local Imports
from caf.brain.ml._functions._ml_inputs import (
    DataClassificationInputs,
    ModellingInputs,
    Models,
    Paths,
    TransformingInputDataInputs,
)
from caf.brain.ml._functions.model_selection.functions import select_model
from caf.brain.ml._functions.process_data_functions.main import (
    main_input_data,
)

LOG = logging.getLogger(__name__)


def main_model_selection(
    data_classification: DataClassificationInputs,
    transforming_inputs: TransformingInputDataInputs,
    modelling: ModellingInputs,
    output: Path,
    paths: Paths,
    train: pd.DataFrame | None = None,
) -> BaseEstimator:
    """
    Function to automatically score and rank algorithms from the Models
    class which can be used in machine learning modelling.

    Parameters
    ----------
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
    paths: Path inputs from the PredictionModelInputs class. These inputs
           define paths to external files. See
           caf/brain/ml/main_models/prediction_model/_ml_inputs.py
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
            modelling=modelling,
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
        if model[0] == Models.XGBOOST_MULTICLASS:
            num_classes = train[data_classification.target_column].nunique()
            selected_model.set_params(num_class=num_classes)
    elif len(model) > 1:
        xgb_selected = any(
            m in (Models.XGBOOST_CLASSIFIER, Models.XGBOOST_MULTICLASS) for m in model
        )
        if xgb_selected:
            LOG.info(
                "XGBoost model selected. This algorithm cannot be compared \n"
                "in the same run as the other provided algorithms. If you \n"
                "want to test the other models, please remove XGBoost from \n"
                "the selection."
            )
            selected_enum = next(
                m for m in model if m in (Models.XGBOOST_CLASSIFIER, Models.XGBOOST_MULTICLASS)
            )
            selected_model = selected_enum.get_model()
            if selected_enum == Models.XGBOOST_MULTICLASS:
                num_classes = train[data_classification.target_column].nunique()
                selected_model.set_params(num_class=num_classes)
        else:
            LOG.info("Beginning model evaluation.")
            selected_model = select_model(
                train=train,
                target_column=data_classification.target_column,
                weight_column=data_classification.weight_column,
                models_to_test=model,
                classification_prediction=transforming_inputs.classification_prediction,
                output_folder=output,
                is_time_series=data_classification.is_time_series,
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

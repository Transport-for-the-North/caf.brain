"""
Created on: 7/11/2025
Original author: Adil Zaheer
"""
import logging
from caf.brain.ml.data_analysis.data_analysis_main import main_evaluate_input_data
from caf.brain.ml.main_models.prediction_model.prediction_model_inputs import PredictionModelInputs
LOG = logging.getLogger(__name__)


def main_test(params: PredictionModelInputs,
              output_path):

    train_transformed, test_transformed = main_evaluate_input_data(
        paths=params.paths,
        data_classification=params.data_classification,
        transforming_inputs=params.transforming_inputs,
        modelling=params.modelling,
        output_path=output_path)

    print(train_transformed)
    print(test_transformed)

    return

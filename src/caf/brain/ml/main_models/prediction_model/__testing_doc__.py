"""
Created on: 7/11/2025
Original author: Adil Zaheer
"""
import logging
from caf.brain.ml.data_analysis.data_analysis_main import main_evaluate_input_data
from caf.brain.ml.feature_selection.feature_selection_main import main_feature_selection
from caf.brain.ml.main_models.prediction_model.prediction_model_inputs import PredictionModelInputs
from caf.brain.ml.model_selection.model_selection_main import main_model_selection

LOG = logging.getLogger(__name__)


def main_test(params: PredictionModelInputs,
              output_path):

    train_final, test_final, cols_dropped_by_feat_select = main_feature_selection(
        paths=params.paths,
        data_classification=params.data_classification,
        transforming_inputs=params.transforming_inputs,
        modelling=params.modelling,
        output=output_path
    )
    print(train_final)
    print(test_final)
    print(cols_dropped_by_feat_select)

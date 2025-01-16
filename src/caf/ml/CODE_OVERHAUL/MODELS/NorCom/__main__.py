# -*- coding: utf-8 -*-
"""
Created on: 1/16/2025
Original author: Adil Zaheer
"""
from caf.ml.CODE_OVERHAUL.hyperparameter_optimisation.hyper_optim_functions import modified_hyper_optimisation
from caf.ml.CODE_OVERHAUL.model_evaluation.evaluation_functions import simple_eval_model
from caf.ml.CODE_OVERHAUL.prediction.prediction_functions import final_prediction, final_prediction_no_validation
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from src.caf.ml.CODE_OVERHAUL.inputs_and_baseclasses.run_inputs import run_file_inputs
from caf.ml.CODE_OVERHAUL.process_data_functions.process_data_main import main_input_data
from caf.ml.CODE_OVERHAUL.model_selection.model_selection_functions import model_prep
import time
import os
import joblib
# TODO NORCOM: make scratches inside repo that can be example runs / run outlines
# TODO NORCOM: add model eval with train data post hyper optim as sometimes validate is None so no accuracy

def main(params: run_file_inputs):
    start_time = time.time()
    train, test, validate = main_input_data(output_path=params.output_path,
                                            file_path=params.file_path,
                                            folder_path=params.folder_path,
                                            target_column=params.target_column,
                                            custom_index=params.custom_index,
                                            column_name_to_drop_rows=params.column_name_to_drop_rows,
                                            value_in_row=params.value_in_row,
                                            weight_column=params.weight_column,
                                            categorical_features=params.categorical_features,
                                            numerical_features=params.numerical_features,
                                            binary_prediction=params.binary_prediction,
                                            time_series_split=params.time_series_split,
                                            validation_path=params.validation_path)

    model, residuals = model_prep(training_df=train,
                                  target_column=params.target_column,
                                  output_folder=params.output_path,
                                  weight_column=params.weight_column,
                                  model_to_use=params.model_choice)

    final_model_filename = os.path.join(params.output_path, 'cafml_final_model.pkl')
    if os.path.exists(final_model_filename):
        print('loading final model')
        final_model = joblib.load(final_model_filename)
    else:
        final_model = modified_hyper_optimisation(model=model,
                                                  data=train,
                                                  target_column=params.target_column,
                                                  output_folder=params.output_path,
                                                  weight_column=params.weight_column,
                                                  original_training_data=train,
                                                  index_columns=params.custom_index)

    if validate is not None:
        y_pred = final_prediction(model=final_model,
                                  data=test,
                                  target_column=params.target_column,
                                  output_folder=params.output_path,
                                  validation=validate,
                                  binary_prediction=params.binary_prediction)

        simple_eval_model(validation_df=validate,
                          y_pred=y_pred,
                          target_column=params.target_column,
                          output_folder=params.output_path)

    else:
        y_pred = final_prediction_no_validation(model=final_model,
                                                data=test,
                                                target_column=params.target_column,
                                                output_folder=params.output_path,
                                                validation=validate,
                                                binary_prediction=params.binary_prediction)

    end_time = time.time()
    print(f"Total run time: {end_time - start_time:.2f} seconds")

    return

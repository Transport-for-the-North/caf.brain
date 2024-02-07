import sys
sys.path.extend(r"C:\Users\Liberty\Documents\GitHub\caf.ml\src")
import warnings


# model specific imports
from pathlib import Path
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from caf.ml.inputs.cafml_inputs import CarInputs2, Models

ALLOWED_MODELS = (ElasticNet, Lasso, Ridge)

from caf.ml.car_access_land_use.process_data_class import DataProcessor
from caf.ml.car_access_land_use.feature_selection import feature_selection_
from caf.ml.functions.feature_selection_gridsearch import (select_model,
                                                           select_param)



def main(params: CarInputs2):
    warnings.filterwarnings("ignore")

    # process data: raw data -> model format
    processed_data = DataProcessor(params.x, params.y, params.folder_path,
                         params.index_columns, params.drop_columns,
                         params.wide_format, params.variable_name,
                         params.value_name, params.outlier_threshold, 
                         params.target_column).data

    print('##########################################')

    if grid_search is None:
        # model format data -> selected features map (30 min run time)
        selected_feature_indices = feature_selection_(processed_data, params.target_column, params.model_type, params.cv_method,
                       params.splits, params.repeats,
                       params.hp_optimisation)
    else:
        x = data.drop(params.target_column, axis=1)
        y = data[params.target_column]

        if params.GS_model_type is None:
            model_name = select_model(x, y)
        else:
            model_name = params.GS_model_type
            print(
                "Performing grid search to determine the best parameters for the model. This can"
                " take some time."
            )
            model_params = select_param(x, y, model_name)

        print("Best params for model determined to be: %s", model_params)
        ml_mod = Models[model_name].value(**model_params)


    #todo (Adil) apply feature map to original dataframe
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    warnings.filterwarnings("default")


if __name__ == "__main__":
    params = CarInputs2(
                        # data paths
                        x=Path(r"E:\TRSE\data\final\sorted_data\data_no_geography\sum_car\2021finalsumcar.csv"),
                        y=None,
                        folder_path=None,

                        # data processing prerequisites
                        # lists
                        index_columns=None,
                        drop_columns=None,

                        # yes or none
                        wide_format=None,
                        # needed if wide format is yes
                        variable_name=None,
                        value_name=None,

                        # target column (to forcast)
                        target_column='sum_cars',

                        # output file path
                        folder=Path(r"E:\caf.ml\data_process_function\car_model_results"),
                        output_folder=Path(r"E:\caf.ml\data_process_function\test_data\output"),

                        # outlier threshold
                        outlier_threshold=None,


                        # feature selection
                        model_type=None,
                        cv_method=None,
                        splits=None,
                        repeats=None,
                        hp_optimisation=None,


    )
    main(params)

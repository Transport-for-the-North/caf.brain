import sys
sys.path.extend(r"C:\Users\Liberty\Documents\GitHub\caf.ml\src")
import warnings


# model specific imports
from pathlib import Path
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from caf.ml.car_access_land_use.inputs import CarInputs2
ALLOWED_MODELS = (ElasticNet, Lasso, Ridge)

from caf.ml.car_access_land_use.process_data_class import DataProcessor
from caf.ml.car_access_land_use.feature_selection import feature_selection_


def main(params: CarInputs2):
    warnings.filterwarnings("ignore")

    # process data: raw data -> model format
    data = DataProcessor(params.x, params.y, params.folder_path,
                         params.index_columns, params.drop_columns,
                         params.wide_format, params.variable_name,
                         params.value_name, params.outlier_threshold, 
                         params.target_column)

    print('##########################################')
    processed_data = data.data
    
    # model format data -> selected features map (30 min run time)
    selected_feature_indices = feature_selection_(processed_data, params.target_column, params.model_type, params.cv_method,
                       params.splits, params.repeats,
                       params.hp_optimisation)

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

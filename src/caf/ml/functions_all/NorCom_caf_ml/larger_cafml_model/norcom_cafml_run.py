import sys
from pathlib import Path
from caf.ml.functions_all.NorCom_caf_ml.larger_cafml_model.norcom_cafml_main import main
from caf.ml.functions_all.NorCom_caf_ml.larger_cafml_model.norcom_inputs import NorCom_inputs

sys.path.extend(r"C:\Users\Liberty\Documents\GitHub\caf.ml\src")


if __name__ == "__main__":
    params = NorCom_inputs(
        # store model
        saved_model=None,
        
        # paths
        x_path=Path(r"E:\2024 work streams\norcom caf.ml integration\norcom_data\0vs1+\cb_tfn_v15_norcom_v3_190624_0_vs_1+.csv"),
        output_folder=Path(r"E:\2024 work streams\norcom caf.ml integration\FINAL\new_functions_testing\fixing model\0vs1"),
        predict_data=Path(r"E:\2024 work streams\norcom caf.ml integration\norcom_data\0vs1+\prediction_data_cb_tfn_v15_v3_norcom_190624_0_vs_1+.csv"),
        validation_data=Path(r"E:\2024 work streams\norcom caf.ml integration\norcom_data\0vs1+\validation_data_cb_tfn_v15_v3_norcom_190624_0_vs_1+.csv"),
        
        # data permutations
        categorical_data='yes',
        numerical_features=None,
        categorical_features=['tfn_at', 'hh_child', 'ns', 'hh_adult'],
        categorical_target='yes',

        # data sorting 
        index_columns=['householdid', 'surveyyear', 'hholdua_b01id'],
        drop_columns=['hh_type', 'hh_income', 'age_b01id', 'hholdnumadults', 'hh_income_band'],
        keep_columns=None,
        target_column='car',

        # model options
        model_type=Models.LOGIT_REGRESSION_ELASTICNET,
        cv_method=None,
        splits=None,
        repeats=None,

        #### PREDICT DATA ####
        single_year_prediction=None,
        multiple_year_prediction='yes',

        # data sorting for prediction_model data
        index_columns_predict=['householdid', 'surveyyear', 'hholdua_b01id'],
        drop_columns_predict=['hh_type', 'hh_income', 'age_b01id', 'hholdnumadults', 'hh_income_band'],
        keep_columns_predict=None,

        # leave as is
        column_name_to_drop_rows=['tfn_at'],
        value_in_row=['20'],
    )
    main(params)

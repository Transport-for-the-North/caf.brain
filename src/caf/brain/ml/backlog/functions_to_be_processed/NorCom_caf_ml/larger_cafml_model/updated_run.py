# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import sys
from pathlib import Path
from caf.ml.backlog.functions_to_be_processed.NorCom_caf_ml.larger_cafml_model.norcom_cafml_main import (
    main,
)
from caf.ml.backlog.functions_to_be_processed.NorCom_caf_ml.larger_cafml_model.norcom_inputs import (
    NorCom_inputs,
    Models,
)

sys.path.extend(r"C:\Users\Liberty\Documents\GitHub\caf.ml\src")


if __name__ == "__main__":
    params = NorCom_inputs(
        classified_build=Path(
            r"E:\2024 work streams\norcom caf.ml integration\FINAL\new_functions_testing\cb_tfn_v15.csv"
        ),
        output_folder=Path(
            r"E:\2024 work streams\norcom caf.ml integration\final_testing_outputs_november"
        ),
        target_column="numcarvan",
        index_columns=["householdid", "surveyyear", "hholdua_b01id"],
        categorical_features=["tfn_at", "hh_child", "ns", "hholdnumadults"],
        numerical_features=None,
        weight_column="w2",
        training_year="2019",
        model_type=Models.LOGIT_REGRESSION_ELASTICNET,
        column_name_to_drop_rows=["tfn_at"],
        value_in_row=["20"],
        binary_prediction="0vs1",
    )

    main(params)

# Built-Ins
from pathlib import Path

# Local Imports
from caf.brain.ml import algorithm_evaluation, hparam_optim
from caf.brain.ml._functions._ml_inputs import Models
from caf.brain.ml._functions.feature_selection.functions import (
    combine_results,
)
from caf.brain.ml._ml import evaluate_data, feature_selection, tidy_data, transform_data


def main():
    path = r"D:\2025 work streams\norcom_mvp+\qa\cb_tfn_v2023.2.csv"
    output = r"D:\2025 work streams\caf.brAIn\prediction_model\final_review_winter25\api_testing_post_review"

    print("starting tidy_data")
    df = tidy_data(
        data_path=Path(path),
        classification_prediction=(2, 0, 3, 1, 5, 7, 4, 6, 9, 8),
        output_path=Path(output),
        categorical_features=["soc", "hh_child"],
        numerical_features=["trips"],
        target="numcarvan",
        custom_index=["householdid"],
    )
    print("tidy_Data output:")
    print(df.head())
    print("finishing tidy_data")

    print("starting transform_data")
    # standalone, just checking if it works
    transform_df = transform_data(
        data=df,
        output_path=Path(output),
        categorical_features=["soc", "hh_child"],
        numerical_features=["trips"],
        target="numcarvan",
    )
    print("transform_data output:")
    print(transform_df.head())
    print("finishing transform_data")

    print("starting evaluate_data")
    train, test = evaluate_data(
        data=df,
        output_path=r"D:\2025 work streams\caf.brAIn\prediction_model\final_review_winter25\eval_data_func",
        categorical_features=["soc", "hh_child"],
        numerical_features=["trips"],
        target="numcarvan",
        classification_prediction=(2, 0, 3, 1, 5, 7, 4, 6, 9, 8),
    )
    print("evaluate_data train output:")
    print(train.head())
    print("evaluate_data test output:")
    print(test.head())
    print("finishing evaluate_data")

    print("starting feature_selection")
    feat_select_train = feature_selection(
        data=train,
        output_path=Path(output),
        categorical_features=["soc", "hh_child"],
        numerical_features=["trips"],
        target="numcarvan",
        is_encoded=True,
    )
    print("feature_selection output:")
    print(feat_select_train.head())
    print("finishing feature_selection")

    print("BONUS: @@@@@@@@@@@@@@ starting combine_results")
    feat_select_test, cols_dropped_by_feat_select = combine_results(
        train_final=feat_select_train,
        target_column="numcarvan",
        weight_column=None,
        test=test,
    )
    print("combine_results output:")
    print(feat_select_test.head())
    print("finishing combine_results")

    print("starting algorithm_evaluation")
    model = algorithm_evaluation(
        data=feat_select_train,
        output_path=Path(output),
        target="numcarvan",
        model_choice=[Models.LOGIT_REGRESSION_ELASTICNET, Models.GRADIENT_BOOSTING_CLASSIFIER],
    )
    print("algorithm_evaluation output:")
    print(model)
    print("finishing algorithm_evaluation")

    print("starting hparam_optim")
    best_model = hparam_optim(
        model_choice=Models.LOGIT_REGRESSION_ELASTICNET,
        data=feat_select_train,
        output_path=Path(output),
        target="numcarvan",
        classification_prediction=(2, 0, 3, 1, 5, 7, 4, 6, 9, 8),
    )
    print("hparam_optim output:")
    print(best_model)
    print("finishing hparam_optim")


if __name__ == "__main__":
    main()

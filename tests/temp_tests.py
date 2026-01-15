import pandas as pd
from sklearn.model_selection import train_test_split

from caf.brain.ml._functions.feature_selection.functions import (
    combine_results,
    rf_feature_selection,
)
from caf.brain.ml._functions.model_selection.functions import select_model
from caf.brain.ml._functions._ml_inputs import PredictionModelInputs, Models
from caf.brain.ml._ml import evaluate_data


def main():
    path = r"D:\2025 work streams\norcom_mvp+\qa\cb_tfn_v2023.2.csv"
    df = pd.read_csv(path)


    df = df[['trips', 'soc', 'hh_child', 'householdid', 'numcarvan']]
    df = df.set_index('householdid')

    train, test = evaluate_data(data=df,
                                output_path=r"D:\2025 work streams\caf.brAIn\prediction_model\final_review_winter25\eval_data_func",
                                categorical_features=['soc', 'hh_child'],
                                numerical_features=['trips'],
                                target='numcarvan',
                                classification_prediction=(2, 0, 3, 1, 5, 7, 4, 6, 9, 8)
                                )

    print(train.head())
    print(train.shape)
    print(test.head())
    print(test.shape)


if __name__ == "__main__":
    main()

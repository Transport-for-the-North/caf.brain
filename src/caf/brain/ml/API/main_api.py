"""
Created on: 9/9/2025
Original author: Adil Zaheer
"""

import logging
import os.path
from pathlib import Path
from typing import Optional
import pandas as pd
from caf.brain.ml import Models
from caf.brain.ml.functions_and_classes.feature_selection.functions import (
    analyse_feature_importance,
)
from caf.brain.ml.functions_and_classes.hparam_optimisation.functions import select_param
from caf.brain.ml.functions_and_classes.model_selection.functions import select_model
from caf.brain.ml.functions_and_classes.process_data_functions.encode_and_scale import (
    process_data_pipeline,
)
from caf.brain.ml.functions_and_classes.process_data_functions.input_data import (
    InitialDataProcessing,
)
from caf.brain.ml.inputs_and_baseclasses.baseclasses import ValidateData

LOG = logging.getLogger(__name__)


class Brain:
    """
    API to use simplified caf.brAIn Machine Learning standalone functions.
    """

    def __init__(
        self,
        data: pd.DataFrame | None,
        data_path: Path | None,
        target: str | None,
        weight: str | None,
        custom_index: list[str] | None,
        categorical_features: list[str] | None,
        numerical_features: list[str] | None,
        classification_prediction: tuple[int, ...] | None,
        output_path: Path | None,
    ):
        self.data = data
        self.data_path = data_path
        self.target = target
        self.weight = weight
        self.custom_index = custom_index
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.classification_prediction = classification_prediction
        self.output_path = output_path

    def _load_data(self) -> pd.DataFrame:
        """
        Loads data into a pandas dataframe.

        Returns
        -------
        Pandas dataframe.
        """
        if self.data is not None:
            return self.data
        if self.data_path:
            return pd.read_csv(self.data_path)
        raise ValueError("No data or data_path provided")

    def validation(self) -> bool:
        """
        Validates data against baseclasses to ensure data is suitable for
        further processing.

        Returns
        -------
        True if all validation checks pass.
        """

        validator = ValidateData(
            dataframe=self.data, custom_index=self.custom_index, target_column=self.target
        )
        validator.index_present()
        validator.target_column_present()
        validator.explanatory_data()
        validator.is_data_numeric()
        validator.data_correct_shape()
        return True

    def check_data(self) -> bool:
        """
        Check if data is suitable for machine learning processes.
        """
        self.validation()
        LOG.info("Data is in correct format for caf.brAIn processes")
        return True

    def tidy_data(
        self,
        column_name_to_drop_rows: Optional[list[str]],
        value_in_row: list[str | float | int] | None,
    ) -> pd.DataFrame:
        """
        Converts semi-structured data into structured inline with machine
        learning standards.

        Parameters
        ----------
        column_name_to_drop_rows: List of string column names that
                                  contain values to drop.
        value_in_row: Corresponding values for column_name_to_drop_rows.

        Returns
        -------
        Structured dataframe.
        """
        dataframe = self._load_data()
        processor = InitialDataProcessing(
            file_path=None,
            folder_path=None,
            output_path=self.output_path,
            target_column=self.target,
            custom_index=self.custom_index,
            column_name_to_drop_rows=column_name_to_drop_rows,
            value_in_row=value_in_row,
            weight_column=None,
            categorical_features=self.categorical_features,
            numerical_features=self.numerical_features,
            classification_prediction=self.classification_prediction,
        )
        processor.df = dataframe
        processed = processor.data_already_split_pipeline(is_test_data=False)
        processed_df = list(processed.values())[0]
        if self.output_path:
            processed_df.to_csv(os.path.join(self.output_path, "tidy_data.csv"))
            LOG.info("Tidy data output as output path provided")
        return processed_df

    def transform_data(
        self,
        sample_size_encode: bool | None = None,
        select_encode_values: bool | None = None,
        encode_values_to_drop: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Encode and or scale data where applicable for machine learning modelling.

        Parameters
        ----------
        sample_size_encode: Optional bool. If true, the data will be split
                        based on sample size. Variables with the largest
                        sample size will be used as reference class.
        select_encode_values: Optional bool. If True, data is split based
                          on custom values set by the user. Corresponds
                          to encode_values_to_drop.
        encode_values_to_drop: If select_encode_values is True, then this
                           must be a list of strings the length of
                           categorical_features. Position one in the list
                           will link to the first variable provided in
                           categorical_features and so on.

        Returns
        -------
        Pandas dataframe of input data encoded and scaled.
        """
        if not self.check_data():
            raise ValueError(
                "Data not suitable for encoding and scaling. Please run \
                              full model flow or tidy_data method prior to \
                              data analysis."
            )
        if not self.output_path:
            raise ValueError(
                "Please provide an output path to use the \
                             _transform_data"
            )

        preprocessed_df, _, _, = process_data_pipeline(
            df=self.data,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            target_column=self.target,
            weight_column=None,
            sample_size_encode=sample_size_encode,
            select_encode_values=select_encode_values,
            encode_values_to_drop=encode_values_to_drop,
            train_encoded=None,
            test_data=False,
            numerical_pipeline=None,
            output_folder=self.output_path,
        )

        preprocessed_df.to_csv(os.path.join(self.output_path, "transformed_data.csv"))
        LOG.info("Transformed data output %s", self.output_path)
        return preprocessed_df

    def data_analysis(self):
        """
        In theory, we want the simplest way possible to do statistical tests
        on your data. This would include linearity, normality etc. In order to
        do these tests, they require residuals based on user training data
        split into x train, x test etc. I think that this function:
        C:\Users\Liberty\Documents\GitHub\caf.brain\src\caf\brain\ml\functions_and_classes\data_analysis\main.py
        already makes it as easy as possible. This function lets the user provide
        very little and creates the models and data they need in order to run the
        tests. All of these other functions wrap those "main" functions but in
        this case I think the main function itself would be the API.

        Apologies if this doesn't make sense, let me know if you have questions
        and I can answer them. Also, not bothered if this stays a class or
        is functions but, I do think these methods for the most part are how id
        expect users to interact with caf.brain in its simplest form.
        
        Returns
        -------

        """
        # todo discuss with ben about this one. lot of functionality we cant get around
        # if not self.check_data():
        #     raise ValueError(
        #         "Data not suitable for data analysis. Please run \
        #                       full model flow or tidy_data method prior to \
        #                       data analysis."
        #     )
        #
        # LOG.warning("Data should already been encoded and scaled where applicable \
        #              If this is not the case please call _transform_data first")
        #
        # train_transformed, test_transformed = pre_forecast_data_analysis(
        #     data_classification=None,
        #     modelling=None,
        #     output_folder=self.output_path,
        #     residuals=None,
        #     model_fit=None,
        #     model_initialised=None,
        #     x_train=None,
        #     x_test=None,
        #     train_scaled=None,
        #     test_scaled=None,
        #     train_unscaled=self.data,
        #     test_unscaled=None,
        #     numerical_pipeline=None,
        # )
        # train_transformed.to_csv(self.output_path / "train_transformed.csv")
        # test_transformed.to_csv(self.output_path / "test_transformed.csv")

        return

    def feat_selection(self, is_encoded: bool = True) -> pd.DataFrame:
        """
        Conduct simple feature selection.

        Parameters
        ----------
        is_encoded: is data encoded and scaled.

        Returns
        -------
        train_final: feature selected train dataset.
        test_final: feature selected test dataset.
        cols_dropped_by_feat_select: data removed due to feature selection.
        """
        if not self.check_data():
            raise ValueError(
                "Data not suitable for data analysis. Please run \
                              full model flow or tidy_data method prior to \
                              data analysis."
            )

        LOG.warning(
            "Data should already been encoded and scaled where applicable \
                     If this is not the case set is_encoded to false"
        )

        if not is_encoded:
            df = self.transform_data()
        else:
            df = self.data

        df_final = analyse_feature_importance(
            train_transformed=df,
            target_column=self.target,
            weight_column=self.weight,
            output_path=self.output_path,
        )
        return df_final

    def algorithim_evaluation(self, model_choice: list[Models]):
        """
        Evaluate which algorithm is best performing. Algorithms must be from
        the Models enum class.

        Parameters
        ----------
        model_choice: List or one algorithm to use as the base of the model.
                    Available algorithms can be seen in
                    ml_inputs.py or __info__.py.
        Returns
        -------
        Initialised best performing model.
        """
        if not self.check_data():
            raise ValueError(
                "Data not suitable for data analysis. Please run \
                              full model flow or tidy_data method prior to \
                              data analysis."
            )
        if not self.output_path:
            raise ValueError(
                "Please provide an output path to use \
                             algorithim_evaluation"
            )

        if not isinstance(model_choice, list):
            model = [model_choice]
        else:
            model = model_choice

        selected_model = select_model(
            train=self.data,
            target_column=self.target,
            weight_column=self.weight,
            models_to_test=model,
            classification_prediction=self.classification_prediction,
            output_folder=self.output_path,
        )

        return selected_model

    def hparam_optim(self, model_choice: Models, is_time_series: bool | None):
        """
        Hyperparamter optimisation for your selected algorithim. Algorithim must
        be part of the Models enum class.

        Parameters
        ----------
        model_choice: List or one algorithm to use as the base of the model.
                    Available algorithms can be seen in
                    ml_inputs.py or __info__.py.
        is_time_series: If true then data must be time series. Time series
                    based characteristics are taken into consideration
                    during function execution.
        Returns
        -------
        Initialised model with the best combination of hyperparameters.
        """
        if not self.output_path:
            raise ValueError(
                "Please provide an output path to use \
                             hparam_optim"
            )

        if not isinstance(model_choice, list):
            model = [model_choice]
        else:
            model = model_choice

        if len(model) > 1:
            LOG.warning(
                "More than one model selected. The first model will be \
            optimised. To find the best performing model, call algorithim_evaluation or \
                        main_model_selection"
            )

        LOG.warning(
            "Data should be encoded and scaled where applicable. Call \
        _transform_data to do this prior to hyperparameter optimisation"
        )

        selected_model = model[0].get_model()

        final_model = select_param(
            train_final=self.data,
            target_column=self.target,
            model_instance=selected_model,
            model_name=model_choice,
            classification_prediction=self.classification_prediction,
            cv=None,
            weight_column=self.weight,
            output_folder=self.output_path,
            is_time_series=is_time_series,
        )

        return final_model


# example usage
brain = Brain(
    data=None,
    data_path=Path("my_data.csv"),
    target="label",
    weight=None,
    custom_index=None,
    categorical_features=["ns-sec", "car_ownership"],
    numerical_features=["age", "income"],
    classification_prediction=[0, 1],
    output_path=Path("/outputs")
)

brain.check_data()

tidy_df = brain.tidy_data(column_name_to_drop_rows=None, value_in_row=None)

transformed_df = brain.transform_data()

selected_df = brain.feat_selection(is_encoded=True)

best_model = brain.algorithim_evaluation(model_choice=[Models.RANDOM_FOREST_REGRESSOR, Models.EXTRA_TREES_REGRESSOR])

optimised_model = brain.hparam_optim(model_choice=Models.RANDOM_FOREST_REGRESSOR, is_time_series=False)

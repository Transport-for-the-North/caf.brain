import numpy as np
import pandas as pd
import pytest
from caf.brain.ml import _ml
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use("Agg")


class TestLoadData:
    def test_load_data_returns_data_when_provided(self):
        """Ensure _load_data returns the provided DataFrame unchanged when data is not None."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _ml._load_data(df, None)
        assert isinstance(result, pd.DataFrame)
        assert result.equals(df)

    def test_load_data_reads_from_path(self, tmp_path):
        """
        Verify _load_data correctly reads a CSV file when data is None and data_path is provided.
        """
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [10, 20, 30]})
        df.to_csv(csv_path, index=False)

        result = _ml._load_data(None, csv_path)

        assert isinstance(result, pd.DataFrame)
        assert result.equals(df)

    def test_load_data_raises_when_no_inputs(self):
        """
        Confirm _load_data raises a ValueError when both data and data_path are None.
        """
        with pytest.raises(ValueError, match="No data or data_path provided"):
            _ml._load_data(None, None)


class TestTidyData:
    def test_tidy_data_uses_provided_dataframe(self, tmp_path):
        """
        Verify that tidy_data returns the provided DataFrame unchanged when data is supplied.
        """
        df = pd.DataFrame({"a": [1, 2, 3]})
        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._functions.process_data_functions.input_data.InitialDataProcessing") as MockProcessor, patch(
                    "caf.brain.ml._functions.process_data_functions.input_data.ValidateData") as MockValidate:

            MockValidate.return_value.validate.return_value = True
            mock_instance = MockProcessor.return_value

            mock_instance.data_already_split_pipeline.return_value = {"processed": df}

            result = _ml.tidy_data(
                data_path=None,
                classification_prediction=None,
                output_path=output,
                categorical_features=None,
                numerical_features=None,
                target="a",
                data=df,
            )

            assert isinstance(result, pd.DataFrame)
            assert result.equals(df)
            assert (output / "tidy_data.csv").exists()

    def test_tidy_data_loads_from_path(self, tmp_path):
        """
        Ensure tidy_data loads a CSV from disk when data_path is provided.
        """
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "feature": [1, 2, 3],
            "a": [10, 20, 30],
        })
        df.to_csv(csv_path, index=False)

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.InitialDataProcessing") as MockProcessor:
            mock_instance = MockProcessor.return_value
            mock_instance.data_already_split_pipeline.return_value = {
                "processed": df
            }

            result = _ml.tidy_data(
                data_path=csv_path,
                classification_prediction=None,
                output_path=output,
                categorical_features=None,
                numerical_features=["feature"],
                target="a",
            )

        assert isinstance(result, pd.DataFrame)
        assert result.equals(df)
        assert (output / "tidy_data.csv").exists()

    def test_tidy_data_passes_correct_arguments_to_processor(self, tmp_path):
        """
        Confirm tidy_data forwards all expected arguments to InitialDataProcessing.
        """
        df = pd.DataFrame({
            "idx": [1, 2],
            "a": [1, 2],
            "cat": ["x", "y"],
            "num": [10, 20],
            "drop_col": ["keep", "drop_val"],
            "w": [1.0, 2.0],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch(
                "caf.brain.ml._ml.InitialDataProcessing"
        ) as MockProcessor:
            mock_instance = MockProcessor.return_value
            mock_instance.data_already_split_pipeline.return_value = {
                "processed": df
            }

            _ml.tidy_data(
                data_path=None,
                classification_prediction=(1, 2),
                output_path=output,
                categorical_features=["cat"],
                numerical_features=["num"],
                target="a",
                custom_index=["idx"],
                weight="w",
                column_name_to_drop_rows=["drop_col"],
                value_in_row=["drop_val"],
                data=df,
            )

            MockProcessor.assert_called_once()

            _, kwargs = MockProcessor.call_args

            assert kwargs["target_column"] == "a"
            assert kwargs["custom_index"] == ["idx"]
            assert kwargs["column_name_to_drop_rows"] == ["drop_col"]
            assert kwargs["value_in_row"] == ["drop_val"]
            assert kwargs["weight_column"] == "w"
            assert kwargs["categorical_features"] == ["cat"]
            assert kwargs["numerical_features"] == ["num"]
            assert kwargs["classification_prediction"] == (1, 2)

    def test_tidy_data_raises_if_no_data(self, tmp_path):
        """
        Check tidy_data raises ValueError when data / data_path is not provided.
        """
        output = tmp_path / "out"

        with pytest.raises(ValueError):
            _ml.tidy_data(
                data_path=None,
                classification_prediction=None,
                output_path=output,
                categorical_features=None,
                numerical_features=None,
                target="a",
                data=None,
            )


class TestTransformData:
    def test_transform_data_numeric_only(self, tmp_path):
        """
        Verify that transform_data uses the numeric only pipeline when
        process_numeric_only=True and returns the mocked transformed DataFrame.
        """
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "target": [0, 1, 0],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch(
                "caf.brain.ml._ml.ValidateData"
        ) as MockValidate, patch(
            "caf.brain.ml._ml._process_data_pipeline_numeric_only"
        ) as MockNumeric:
            MockValidate.return_value.validate.return_value = True

            processed_df = pd.DataFrame({
                "num_scaled": [0.1, 0.2, 0.3]
            })
            MockNumeric.return_value = processed_df

            result = _ml.transform_data(
                data=df,
                output_path=output,
                categorical_features=None,
                numerical_features=["num"],
                target="target",
                process_numeric_only=True,
            )

        assert result.equals(processed_df)
        assert (output / "transformed_data.csv").exists()
        MockNumeric.assert_called_once()

    def test_transform_data_categorical_only(self, tmp_path):
        """
        Ensure transform_data uses the categorical only pipeline when
        process_categorical_only=True and returns the mocked transformed DataFrame.
        """
        df = pd.DataFrame({
            "cat": ["a", "b", "a"],
            "target": [1, 0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch(
                "caf.brain.ml._ml.ValidateData"
        ) as MockValidate, patch(
            "caf.brain.ml._ml._process_data_pipeline_categorical_only"
        ) as MockCat:
            MockValidate.return_value.validate.return_value = True

            processed_df = pd.DataFrame({
                "cat_encoded": [0, 1, 0]
            })
            MockCat.return_value = processed_df

            result = _ml.transform_data(
                data=df,
                output_path=output,
                categorical_features=["cat"],
                numerical_features=None,
                target="target",
                process_categorical_only=True,
            )

        assert result.equals(processed_df)
        assert (output / "transformed_data.csv").exists()
        MockCat.assert_called_once()

    def test_transform_data_full_pipeline(self, tmp_path):
        """
        Confirm transform_data calls the full pipeline when numeric only
        or categorical only flags are not set, returning the mocked transformed DataFrame.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "cat": ["x", "y"],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch(
                "caf.brain.ml._ml.ValidateData"
        ) as MockValidate, patch(
            "caf.brain.ml._ml.process_data_pipeline"
        ) as MockPipeline:
            MockValidate.return_value.validate.return_value = True

            processed_df = pd.DataFrame({
                "num_scaled": [0.1, 0.2],
                "cat_encoded": [0, 1],
            })

            MockPipeline.return_value = (
                processed_df,
                None,
                None,
            )

            result = _ml.transform_data(
                data=df,
                output_path=output,
                categorical_features=["cat"],
                numerical_features=["num"],
                target="target",
            )

        assert result.equals(processed_df)
        assert (output / "transformed_data.csv").exists()
        MockPipeline.assert_called_once()

    def test_transform_data_invalid_data(self, tmp_path):
        """
        Check transform_data raises ValueError when ValidateData.validate() returns False.
        """
        df = pd.DataFrame({"a": [1, 2]})
        output = tmp_path / "out"
        output.mkdir()

        with patch(
                "caf.brain.ml._functions.process_data_functions.input_data.ValidateData"
        ) as MockValidate:
            MockValidate.return_value.validate.return_value = False

            with pytest.raises(ValueError):
                _ml.transform_data(
                    data=df,
                    output_path=output,
                    categorical_features=None,
                    numerical_features=None,
                    target="missing",
                )


class TestFeatureSelection:
    def test_feature_selection_encoded_no_test_data(self, tmp_path):
        """
        Verify that feature_selection returns the feature‑selected training DataFrame
        when data is already encoded and no test_data is provided.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate, patch("caf.brain.ml._ml.analyse_feature_importance") as MockAnalyse:

            MockValidate.return_value.validate.return_value = True

            processed_df = pd.DataFrame({"num": [1, 2], "target": [0, 1]})
            MockAnalyse.return_value = processed_df

            result_train, result_test = _ml.feature_selection(
                data=df,
                output_path=output,
                categorical_features=None,
                numerical_features=["num"],
                target="target",
                is_encoded=True,
            )

        assert isinstance(result_train, pd.DataFrame)
        assert result_train.equals(processed_df)
        assert result_test is None
        MockAnalyse.assert_called_once()

    def test_feature_selection_not_encoded_calls_transform_data(self, tmp_path):
        """
        Ensure feature_selection calls transform_data when is_encoded=False,
        and returns the feature‑selected transformed DataFrame.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate, patch("caf.brain.ml._ml.transform_data") as MockTransform, patch("caf.brain.ml._ml.analyse_feature_importance") as MockAnalyse:
            MockValidate.return_value.validate.return_value = True

            transformed_df = pd.DataFrame({"num_scaled": [0.1, 0.2], "target": [0, 1]})
            MockTransform.return_value = transformed_df

            analysed_df = pd.DataFrame({"num_scaled": [0.1, 0.2]})
            MockAnalyse.return_value = analysed_df

            result_train, result_test = _ml.feature_selection(
                data=df,
                output_path=output,
                categorical_features=None,
                numerical_features=["num"],
                target="target",
                is_encoded=False,
            )

        MockTransform.assert_called_once()
        MockAnalyse.assert_called_once()
        assert result_train.equals(analysed_df)
        assert result_test is None

    def test_feature_selection_with_test_data(self, tmp_path):
        """
        Confirm feature_selection applies combine_results when test_data is provided,
        returning both feature‑selected training data and combined test data.
        """
        train_df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        test_df = pd.DataFrame({
            "num": [5, 6],
            "target": [1, 0],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate, patch("caf.brain.ml._ml.analyse_feature_importance") as MockAnalyse, patch("caf.brain.ml._ml.combine_results") as MockCombine:
            MockValidate.return_value.validate.return_value = True

            analysed_df = pd.DataFrame({"num": [1, 2]})
            MockAnalyse.return_value = analysed_df

            combined_test_df = pd.DataFrame({"num": [5, 6]})
            MockCombine.return_value = (combined_test_df, None)

            result_train, result_test = _ml.feature_selection(
                data=train_df,
                output_path=output,
                categorical_features=None,
                numerical_features=["num"],
                target="target",
                test_data=test_df,
            )

        MockAnalyse.assert_called_once()
        MockCombine.assert_called_once()
        assert result_train.equals(analysed_df)
        assert result_test.equals(combined_test_df)

    def test_feature_selection_invalid_data(self, tmp_path):
        """
        Check feature_selection raises ValueError when ValidateData.validate() returns False.
        """
        df = pd.DataFrame({"a": [1, 2]})
        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate:
            MockValidate.return_value.validate.return_value = False

            with pytest.raises(ValueError):
                _ml.feature_selection(
                    data=df,
                    output_path=output,
                    categorical_features=None,
                    numerical_features=None,
                    target="missing",
                )


class TestAlgorithmEvaluation:
    def test_algorithm_evaluation_basic(self, tmp_path):
        """
        Verify that algorithm_evaluation returns the selected model when data is valid
        and a single model_choice is provided.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate, patch("caf.brain.ml._ml.select_model") as MockSelect:

            MockValidate.return_value.validate.return_value = True

            mock_model = MagicMock()
            MockSelect.return_value = mock_model

            result = _ml.algorithm_evaluation(
                model_choice=_ml.Models.LOGIT_REGRESSION_L1,
                data=df,
                output_path=output,
                target="target",
            )

        assert result is mock_model
        MockSelect.assert_called_once()

    def test_algorithm_evaluation_list_of_models(self, tmp_path):
        """
        Ensure algorithm_evaluation accepts a list of models and forwards the list
        unchanged to select_model.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate, patch("caf.brain.ml._ml.select_model") as MockSelect:

            MockValidate.return_value.validate.return_value = True

            mock_model = MagicMock()
            MockSelect.return_value = mock_model

            result = _ml.algorithm_evaluation(
                model_choice=[
                    _ml.Models.LOGIT_REGRESSION_L1,
                    _ml.Models.RANDOM_FOREST_REGRESSOR,
                ],
                data=df,
                output_path=output,
                target="target",
            )

        assert result is mock_model
        MockSelect.assert_called_once()
        # list was passed through unchanged
        _, kwargs = MockSelect.call_args
        assert kwargs["models_to_test"] == [
            _ml.Models.LOGIT_REGRESSION_L1,
            _ml.Models.RANDOM_FOREST_REGRESSOR,
        ]

    def test_algorithm_evaluation_xgboost_preparation_called(self, tmp_path):
        """
        Confirm that XGBoost models trigger xgboost_preparation before model selection.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate, \
             patch("caf.brain.ml._ml.InitialDataProcessing.xgboost_preparation") as MockPrep, \
             patch("caf.brain.ml._ml.select_model") as MockSelect:

            MockValidate.return_value.validate.return_value = True

            prepared_df = pd.DataFrame({"num": [10, 20], "target": [0, 1]})
            MockPrep.return_value = prepared_df

            mock_model = MagicMock()
            MockSelect.return_value = mock_model

            result = _ml.algorithm_evaluation(
                model_choice=_ml.Models.XGBOOST_CLASSIFIER,
                data=df,
                output_path=output,
                target="target",
                classification_prediction=(0, 1),
            )

        MockPrep.assert_called_once()
        MockSelect.assert_called_once()
        assert result is mock_model

    def test_algorithm_evaluation_invalid_data(self, tmp_path):
        """
        Check algorithm_evaluation raises ValueError when ValidateData.validate() fails.
        """
        df = pd.DataFrame({"a": [1, 2]})
        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate:
            MockValidate.return_value.validate.return_value = False

            with pytest.raises(ValueError):
                _ml.algorithm_evaluation(
                    model_choice=_ml.Models.LOGIT_REGRESSION_L1,
                    data=df,
                    output_path=output,
                    target="missing",
                )

    def test_algorithm_evaluation_single_model_wrapped_into_list(self, tmp_path):
        """
        Ensure a single model_choice is wrapped into a list before being passed to select_model.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate, patch("caf.brain.ml._ml.select_model") as MockSelect:

            MockValidate.return_value.validate.return_value = True

            mock_model = MagicMock()
            MockSelect.return_value = mock_model

            result = _ml.algorithm_evaluation(
                model_choice=_ml.Models.RANDOM_FOREST_REGRESSOR,
                data=df,
                output_path=output,
                target="target",
            )

        _, kwargs = MockSelect.call_args
        assert kwargs["models_to_test"] == [_ml.Models.RANDOM_FOREST_REGRESSOR]
        assert result is mock_model


class TestHparamOptim:
    def test_hparam_optim_basic(self, tmp_path):
        """
        Verify that hparam_optim returns the model produced by select_param
        when a single model_choice is provided.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.select_param") as MockSelect:
            mock_model = MagicMock()
            MockSelect.return_value = mock_model

            result = _ml.hparam_optim(
                model_choice=_ml.Models.LOGIT_REGRESSION_L1,
                data=df,
                output_path=output,
                target="target",
            )

        assert result is mock_model
        MockSelect.assert_called_once()

    def test_hparam_optim_xgboost_preparation_called(self, tmp_path):
        """
        Ensure XGBoost models trigger xgboost_preparation before select_param is called.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.InitialDataProcessing.xgboost_preparation") as MockPrep, patch("caf.brain.ml._ml.select_param") as MockSelect:

            prepared_df = pd.DataFrame({"num": [10, 20], "target": [0, 1]})
            MockPrep.return_value = prepared_df

            mock_model = MagicMock()
            MockSelect.return_value = mock_model

            result = _ml.hparam_optim(
                model_choice=_ml.Models.XGBOOST_CLASSIFIER,
                data=df,
                output_path=output,
                target="target",
                classification_prediction=(0, 1),
            )

        MockPrep.assert_called_once()
        MockSelect.assert_called_once()
        assert result is mock_model

    def test_hparam_optim_multiple_models_warns_and_uses_first(self, tmp_path):
        """
        Confirm that when multiple models are provided, hparam_optim warns and
        only uses the first model for optimisation.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.select_param") as MockSelect, patch("warnings.warn") as MockWarn:

            mock_model = MagicMock()
            MockSelect.return_value = mock_model

            result = _ml.hparam_optim(
                model_choice=[
                    _ml.Models.RANDOM_FOREST_REGRESSOR,
                    _ml.Models.LOGIT_REGRESSION_L1,
                ],
                data=df,
                output_path=output,
                target="target",
            )

        # first model should be used
        _, kwargs = MockSelect.call_args
        assert isinstance(
            kwargs["model_instance"],
            type(_ml.Models.RANDOM_FOREST_REGRESSOR.get_model()),
        )
        assert result is mock_model
        MockWarn.assert_called()

    def test_hparam_optim_missing_output_path_raises(self):
        """
        Check hparam_optim raises ValueError when output_path is missing.
        """
        df = pd.DataFrame({"num": [1, 2], "target": [0, 1]})

        with pytest.raises(ValueError):
            _ml.hparam_optim(
                model_choice=_ml.Models.LOGIT_REGRESSION_L1,
                data=df,
                output_path=None,
                target="target",
            )


class TestEvaluateData:
    def test_evaluate_data_basic_regression(self, tmp_path):
        """
        Verify that evaluate_data executes the full regression pipeline:
        unscaled split, scaling, train/test split, model initialisation,
        and forecast analysis.
        """
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "target": [10, 20, 30],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate, \
             patch("caf.brain.ml._ml.simple_data_split") as MockSplit, \
             patch("caf.brain.ml._ml.process_data_pipeline") as MockPipeline, \
             patch("caf.brain.ml._ml.simple_train_test_split") as MockTrainTest, \
             patch("caf.brain.ml._ml.initialise_model") as MockInitModel, \
             patch("caf.brain.ml._ml.pre_forecast_data_analysis") as MockAnalysis:

            MockValidate.return_value.validate.return_value = True

            # split unscaled
            train_unscaled = pd.DataFrame({"num": [1, 2], "target": [10, 20]})
            test_unscaled = pd.DataFrame({"num": [3], "target": [30]})
            MockSplit.return_value = (train_unscaled, test_unscaled, None)

            # pipeline for train
            train_scaled = pd.DataFrame({"num_scaled": [0.1, 0.2], "target": [10, 20]})
            pipeline_out = MagicMock()
            MockPipeline.side_effect = [
                (train_scaled, None, pipeline_out),  # train
                (pd.DataFrame({"num_scaled": [0.3], "target": [30]}), None, None),  # test
            ]

            # train/test split
            x_train = pd.DataFrame({"num_scaled": [0.1, 0.2]})
            x_test = pd.DataFrame({"num_scaled": [0.3]})
            y_train = pd.Series([10, 20])
            y_test = pd.Series([30])
            x_train_weight = None
            MockTrainTest.return_value = (x_train, x_test, y_train, y_test, x_train_weight)

            # model fitting
            model_fit = MagicMock()
            residuals = MagicMock()
            MockInitModel.return_value = (model_fit, residuals, None)

            _ml.evaluate_data(data=df, output_path=output, categorical_features=None, numerical_features=["num"], target="target")

        MockSplit.assert_called_once()
        assert MockPipeline.call_count == 2
        MockTrainTest.assert_called_once()
        MockInitModel.assert_called_once()
        MockAnalysis.assert_called_once()

    def test_evaluate_data_classification_uses_logistic_regression(self, tmp_path):
        """
        Ensure evaluate_data selects the classification branch and passes
        classification_prediction to initialise_model.
        """
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "target": [0, 1, 0],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate, \
             patch("caf.brain.ml._ml.simple_data_split") as MockSplit, \
             patch("caf.brain.ml._ml.process_data_pipeline") as MockPipeline, \
             patch("caf.brain.ml._ml.simple_train_test_split") as MockTrainTest, \
             patch("caf.brain.ml._ml.initialise_model") as MockInitModel:

            MockValidate.return_value.validate.return_value = True

            MockSplit.return_value = (df, df, None)
            MockPipeline.return_value = (df, None, None)
            MockTrainTest.return_value = (df, df, df["target"], df["target"], None)

            MockInitModel.return_value = (MagicMock(), MagicMock(), None)

            _ml.evaluate_data(
                data=df,
                output_path=output,
                categorical_features=None,
                numerical_features=["num"],
                target="target",
                classification_prediction=(0, 1),
            )

        # logistic regression branch should be used
        _, kwargs = MockInitModel.call_args
        assert kwargs["classification_prediction"] == (0, 1)

    def test_evaluate_data_invalid_data(self, tmp_path):
        """
        Confirm evaluate_data raises ValueError when ValidateData.validate()
        indicates the input data is unsuitable.
        """
        df = pd.DataFrame({"a": [1, 2]})
        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.ValidateData") as MockValidate:
            MockValidate.return_value.validate.return_value = False

            with pytest.raises(ValueError):
                _ml.evaluate_data(
                    data=df,
                    output_path=output,
                    categorical_features=None,
                    numerical_features=None,
                    target="missing",
                )


class TestSimpleDataSplit:
    def test_split_by_value_calls_split_by_column_value(self, tmp_path):
        """
        Verify that simple_data_split delegates to split_by_column_value when
        split_by_value is provided and returns the expected train, test, and
        validate DataFrames.
        """
        df = pd.DataFrame({
            "year": [2018, 2019, 2020],
            "target": [1, 0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.split_by_column_value") as MockSplit:
            train_df = pd.DataFrame({"year": [2018], "target": [1]})
            test_df = pd.DataFrame({"year": [2019], "target": [0]})
            validate_df = pd.DataFrame({"year": [2020], "target": [1]})

            MockSplit.return_value = (train_df, test_df, validate_df)

            train, test, validate = _ml.simple_data_split(
                data=df,
                target="target",
                output_path=output,
                split_by_value="2019",
                custom_index=["year"],
            )

        MockSplit.assert_called_once()
        assert train is not None
        assert test is not None
        assert validate is not None

        assert train.equals(train_df)
        assert test.equals(test_df)
        assert validate.equals(validate_df)

    def test_split_by_value_without_custom_index_raises(self, tmp_path):
        """
        Confirm simple_data_split raises ValueError when split_by_value is set
        but custom_index is missing.
        """
        df = pd.DataFrame({"year": [2018, 2019], "target": [1, 0]})
        output = tmp_path / "out"
        output.mkdir()

        with pytest.raises(ValueError):
            _ml.simple_data_split(
                data=df,
                target="target",
                output_path=output,
                split_by_value="2019",
                custom_index=None,
            )

    def test_classification_branch_calls_stratified_split(self, tmp_path):
        """
        Ensure simple_data_split calls stratified_split_with_categories when
        classification_prediction is provided.
        """
        df = pd.DataFrame({
            "cat": ["a", "b", "a"],
            "target": [1, 0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        with patch("caf.brain.ml._ml.stratified_split_with_categories") as MockStrat:
            train_df = pd.DataFrame({"cat": ["a"], "target": [1]})
            test_df = pd.DataFrame({"cat": ["b"], "target": [0]})
            validate_df = pd.DataFrame({"cat": ["a"], "target": [1]})

            MockStrat.return_value = (train_df, test_df, validate_df)

            train, test, validate = _ml.simple_data_split(
                data=df,
                target="target",
                output_path=output,
                classification_prediction=(0, 1),
                categorical_features=["cat"],
            )

        MockStrat.assert_called_once()
        assert train is not None
        assert test is not None
        assert validate is not None
        assert train.equals(train_df)
        assert test.equals(test_df)
        assert validate.equals(validate_df)

    def test_simple_split_creates_train_test_and_validate_csv(self, tmp_path):
        """
        Verify that simple_data_split performs a standard train/test split and
        writes train.csv, test.csv, and validate.csv to disk.
        """
        df = pd.DataFrame({
            "num": [1, 2, 3, 4, 5],
            "target": [10, 20, 30, 40, 50],
        })

        output = tmp_path / "out"
        output.mkdir()

        _ = _ml.simple_data_split(
            data=df,
            target="target",
            output_path=output,
        )

        assert (output / "train.csv").exists()
        assert (output / "test.csv").exists()
        assert (output / "validate.csv").exists()
        validate_loaded = pd.read_csv(output / "validate.csv", index_col=0)
        assert list(validate_loaded.columns) == ["target"]

    def test_simple_split_drops_weight_column(self, tmp_path):
        """
        Ensure simple_data_split removes the weight column from the test set and
        produces a validate DataFrame containing only the target column.
        """
        df = pd.DataFrame({
            "num": [1, 2, 3, 4, 5],
            "target": [10, 20, 30, 40, 50],
            "w": [0.1, 0.2, 0.3, 0.4, 0.5],
        })

        output = tmp_path / "out"
        output.mkdir()

        train, test, validate = _ml.simple_data_split(
            data=df,
            target="target",
            output_path=output,
            weight="w",
        )

        assert train is not None
        assert test is not None
        assert validate is not None
        assert "w" not in test.columns
        assert list(validate.columns) == ["target"]


class TestSimplePrediction:
    def test_simple_prediction_regression_no_validation(self, tmp_path):
        """
        Verify that simple_prediction performs a regression prediction without validation
        and writes both final_predictions.csv and final_model_coefficients.csv.
        """
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "target": [10, 20, 30],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict.return_value = np.array([11, 21, 31])

        with patch("caf.brain.ml._ml.calculate_final_coefficients") as MockCoeff:

            MockCoeff.return_value = pd.DataFrame({"coef": [1.0]})

            _ml.simple_prediction(
                model=model,
                test=df,
                target_column="target",
                output_folder=output,
            )

        assert (output / "final_predictions.csv").exists()
        assert (output / "final_model_coefficients.csv").exists()
        model.predict.assert_called_once()

    def test_simple_prediction_regression_with_validation(self, tmp_path):
        """
        Ensure regression predictions with validation produce model_performance.csv
        and final_predictions.csv, and call calculate_final_coefficients once.
        """
        test_df = pd.DataFrame({
            "num": [1, 2],
            "target": [10, 20],
        })

        validation_df = pd.DataFrame({
            "num": [1, 2],
            "target": [12, 22],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict.return_value = np.array([11, 21])

        with patch("caf.brain.ml._ml.calculate_final_coefficients") as MockCoeff:
            MockCoeff.return_value = pd.DataFrame({"coef": [1.0]})

            _ml.simple_prediction(
                model=model,
                test=test_df,
                target_column="target",
                output_folder=output,
                validation=validation_df,
            )

        assert (output / "model_performance.csv").exists()
        assert (output / "final_predictions.csv").exists()
        MockCoeff.assert_called_once()

    def test_simple_prediction_classification_no_validation(self, tmp_path):
        """
        Verify classification predictions without validation use predict_proba
        and produce final_predictions.csv.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        model.classes_ = np.array([0, 1])

        with patch("caf.brain.ml._ml.calculate_final_coefficients") as MockCoeff:
            MockCoeff.return_value = pd.DataFrame({"coef": [1.0]})

            _ml.simple_prediction(
                model=model,
                test=df,
                target_column="target",
                output_folder=output,
                classification_prediction=(0, 1),
            )

        assert (output / "final_predictions.csv").exists()
        model.predict_proba.assert_called_once()

    def test_simple_prediction_classification_with_validation(self, tmp_path):
        """
        Ensure classification predictions with validation produce model_performance.csv
        and final_predictions.csv.
        """
        test_df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        validation_df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
        model.classes_ = np.array([0, 1])

        with patch("caf.brain.ml._ml.calculate_final_coefficients") as MockCoeff:
            MockCoeff.return_value = pd.DataFrame({"coef": [1.0]})

            _ml.simple_prediction(
                model=model,
                test=test_df,
                target_column="target",
                output_folder=output,
                validation=validation_df,
                classification_prediction=(0, 1),
            )

        assert (output / "model_performance.csv").exists()
        assert (output / "final_predictions.csv").exists()

    def test_simple_prediction_xgb_mapping(self, tmp_path):
        """
        Confirm XGBClassifierBinary/Multiclass predictions are mapped using
        classification_prediction and written correctly to final_predictions.csv.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        class FakeXGB(_ml.XGBClassifierBinary):
            @property
            def classes_(self):
                return np.array([0, 1])

        model = FakeXGB()
        model.predict_proba = MagicMock(
            return_value=np.array([[0.1, 0.9], [0.8, 0.2]])
        )

        with patch("caf.brain.ml._ml.calculate_final_coefficients") as MockCoeff:
            MockCoeff.return_value = pd.DataFrame({"coef": [1.0]})

            _ml.simple_prediction(
                model=model,
                test=df,
                target_column="target",
                output_folder=output,
                classification_prediction=(10, 20),
            )

        preds = pd.read_csv(output / "final_predictions.csv")[
            "predicted_target_column"
        ].tolist()

        assert preds == [20, 10]

    def test_simple_prediction_linear_svc(self, tmp_path):
        """
        Verify LinearSVC classification uses predict() and produces final_predictions.csv.
        """
        df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock(spec=_ml.LinearSVC)
        model.predict.return_value = np.array([0, 1])

        with patch("caf.brain.ml._ml.calculate_final_coefficients") as MockCoeff:
            MockCoeff.return_value = pd.DataFrame({"coef": [1.0]})

            _ml.simple_prediction(
                model=model,
                test=df,
                target_column="target",
                output_folder=output,
                classification_prediction=(0, 1),
            )

        model.predict.assert_called_once()

    def test_simple_prediction_missing_target_column_with_validation_raises(self, tmp_path):
        """
        Confirm simple_prediction raises ValueError when validation is provided
        but target_column is empty.
        """
        df = pd.DataFrame({"num": [1, 2]})
        validation_df = pd.DataFrame({"num": [1, 2], "target": [10, 20]})

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()

        with pytest.raises(ValueError):
            _ml.simple_prediction(
                model=model,
                test=df,
                target_column="",
                output_folder=output,
                validation=validation_df,
            )


class TestVisualiseModelPerformance:
    def test_regression_outputs_created(self, tmp_path):
        """
        Verify that regression visualisation produces metrics.csv and all expected
        regression plots (predicted vs actual, residuals, residual distribution).
        """
        test_df = pd.DataFrame({
            "num": [1, 2, 3],
            "target": [10, 20, 30],
        })
        validation_df = pd.DataFrame({
            "num": [1, 2, 3],
            "target": [12, 22, 32],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict.return_value = np.array([11, 21, 31])

        _ml.visualise_model_performance(
            model=model,
            test=test_df,
            target="target",
            output_folder=output,
            validation=validation_df,
            is_classification=False,
        )

        assert (output / "metrics.csv").exists()
        assert (output / "pred_vs_actual_test.png").exists()
        assert (output / "residuals_test.png").exists()
        assert (output / "residual_distribution_test.png").exists()

        model.predict.assert_called_once()

    def test_classification_outputs_created(self, tmp_path):
        """
        Ensure classification visualisation produces metrics.csv and a confusion
        matrix plot when predict_proba is available.
        """
        test_df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })
        validation_df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict.return_value = np.array([0, 1])
        model.classes_ = np.array([0, 1])

        model.predict_proba.return_value = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
        ])

        _ml.visualise_model_performance(
            model=model,
            test=test_df,
            target="target",
            output_folder=output,
            validation=validation_df,
            is_classification=True,
        )

        assert (output / "metrics.csv").exists()
        assert (output / "confusion_matrix_test.png").exists()

    def test_classification_with_predict_proba_creates_roc_and_pr(self, tmp_path):
        """
        Confirm that binary classification with predict_proba produces ROC and
        precision–recall curve plots.
        """
        test_df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })
        validation_df = pd.DataFrame({
            "num": [1, 2],
            "target": [0, 1],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict.return_value = np.array([0, 1])
        model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        model.classes_ = np.array([0, 1])

        _ml.visualise_model_performance(
            model=model,
            test=test_df,
            target="target",
            output_folder=output,
            validation=validation_df,
            is_classification=True,
        )

        assert (output / "roc_curve_test.png").exists()
        assert (output / "precision_recall_test.png").exists()

    def test_model_loaded_from_path(self, tmp_path):
        """
        Verify that visualise_model_performance loads a model from the given file path
        when model is provided as a string.
        """
        test_df = pd.DataFrame({
            "num": [1, 2],
            "target": [10, 20],
        })
        validation_df = pd.DataFrame({
            "num": [1, 2],
            "target": [12, 22],
        })

        output = tmp_path / "out"
        output.mkdir()

        model_path = tmp_path / "model.pkl"

        with patch("caf.brain.ml._ml.joblib.load") as MockLoad:
            fake_model = MagicMock()
            fake_model.predict.return_value = np.array([11, 21])
            MockLoad.return_value = fake_model

            _ml.visualise_model_performance(
                model=str(model_path),
                test=test_df,
                target="target",
                output_folder=output,
                validation=validation_df,
                is_classification=False,
            )

        MockLoad.assert_called_once()

    def test_test_and_validation_loaded_from_path(self, tmp_path):
        """
        Ensure test and validation datasets are loaded from CSV paths when
        provided as strings.
        """
        test_path = tmp_path / "test.csv"
        validation_path = tmp_path / "validation.csv"

        test_df = pd.DataFrame({"num": [1], "target": [10]})
        validation_df = pd.DataFrame({"num": [1], "target": [12]})

        test_df.to_csv(test_path, index=False)
        validation_df.to_csv(validation_path, index=False)

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict.return_value = np.array([11])

        _ml.visualise_model_performance(
            model=model,
            test=str(test_path),
            target="target",
            output_folder=output,
            validation=str(validation_path),
            is_classification=False,
        )

        assert (output / "metrics.csv").exists()

    def test_index_columns_are_applied(self, tmp_path):
        """
        Verify that index_columns are applied to both test and validation data
        before prediction and metric calculation.
        """
        test_df = pd.DataFrame({
            "year": [2020, 2021],
            "num": [1, 2],
            "target": [10, 20],
        })
        validation_df = pd.DataFrame({
            "year": [2020, 2021],
            "num": [1, 2],
            "target": [12, 22],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict.return_value = np.array([11, 21])

        _ml.visualise_model_performance(
            model=model,
            test=test_df,
            target="target",
            output_folder=output,
            validation=validation_df,
            index_columns=["year"],
            is_classification=False,
        )

        assert (output / "metrics.csv").exists()
        assert model.predict.called

    def test_weight_column_used(self, tmp_path):
        """
        Confirm that weight_column is used when computing regression metrics.
        """
        test_df = pd.DataFrame({
            "num": [1, 2],
            "target": [10, 20],
            "w": [0.5, 1.0],
        })
        validation_df = pd.DataFrame({
            "num": [1, 2],
            "target": [12, 22],
        })

        output = tmp_path / "out"
        output.mkdir()

        model = MagicMock()
        model.predict.return_value = np.array([11, 21])

        _ml.visualise_model_performance(
            model=model,
            test=test_df,
            target="target",
            output_folder=output,
            validation=validation_df,
            weight_column="w",
            is_classification=False,
        )
        assert (output / "metrics.csv").exists()


class TestDependantVariableTesting:
    def test_continuous_variable_outputs_created(self, tmp_path):
        """
        Verify that continuous variables produce summary statistics and
        distribution plots when the number of unique values exceeds the
        threshold.
        """
        df = pd.DataFrame({
            "target": list(range(1, 16)),
        })

        output = tmp_path / "out"
        output.mkdir()

        _ml.dependant_variable_testing(
            data=df,
            target="target",
            output_folder=output,
        )

        assert (output / "continuous_summary.csv").exists()
        assert (output / "target_hist.png").exists()
        assert (output / "target_boxplot.png").exists()

    def test_continuous_variable_with_weight_outputs_weighted_stats(self, tmp_path):
        """
        Ensure continuous variables with a weight column produce both the
        summary statistics file and weighted_stats.csv.
        """
        df = pd.DataFrame({
            "target": list(range(1, 16)),
            "w": [float(i) for i in range(1, 16)],
        })

        output = tmp_path / "out"
        output.mkdir()

        _ml.dependant_variable_testing(
            data=df,
            target="target",
            output_folder=output,
            weight_column="w",
        )

        assert (output / "continuous_summary.csv").exists()
        assert (output / "weighted_stats.csv").exists()

    def test_categorical_variable_outputs_created(self, tmp_path):
        """
        Confirm categorical variables with a weight column produce both
        categorical_counts.csv and categorical_weighted_counts.csv.
        """
        df = pd.DataFrame({
            "target": ["a", "b", "a", "c"],
        })

        output = tmp_path / "out"
        output.mkdir()

        _ml.dependant_variable_testing(
            data=df,
            target="target",
            output_folder=output,
        )

        assert (output / "categorical_counts.csv").exists()
        assert (output / "target_countplot.png").exists()

    def test_categorical_variable_with_weight_outputs_weighted_counts(self, tmp_path):
        """
        Confirm that categorical variables with a weight column produce both
        categorical_counts.csv and categorical_weighted_counts.csv.
        """
        df = pd.DataFrame({
            "target": ["a", "b", "a"],
            "w": [1.0, 2.0, 3.0],
        })

        output = tmp_path / "out"
        output.mkdir()

        _ml.dependant_variable_testing(
            data=df,
            target="target",
            output_folder=output,
            weight_column="w",
        )

        assert (output / "categorical_counts.csv").exists()
        assert (output / "categorical_weighted_counts.csv").exists()

    def test_data_loaded_from_path(self, tmp_path):
        """
        Ensure dependant_variable_testing loads data correctly when a CSV path
        is provided instead of a DataFrame.
        """
        df = pd.DataFrame({
            "target": list(range(1, 16)),
        })

        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        output = tmp_path / "out"
        output.mkdir()

        _ml.dependant_variable_testing(
            data=str(csv_path),
            target="target",
            output_folder=output,
        )

        assert (output / "continuous_summary.csv").exists()

    def test_missing_target_column_raises(self, tmp_path):
        """
        Confirm dependant_variable_testing raises ValueError when the target
        column is missing from the input data.
        """
        df = pd.DataFrame({"x": [1, 2, 3]})

        output = tmp_path / "out"
        output.mkdir()

        with pytest.raises(ValueError):
            _ml.dependant_variable_testing(
                data=df,
                target="missing",
                output_folder=output,
            )

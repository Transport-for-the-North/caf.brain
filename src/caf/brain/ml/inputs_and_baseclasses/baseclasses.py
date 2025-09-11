"""
Created on: 12/13/2024
Original author: Adil Zaheer
"""

# Built-Ins
import warnings
from abc import ABC, abstractmethod
from typing import Union

# Third Party
import pandas as pd


# # # CLASSES # # #
class BaseDataClass(ABC):
    """
    Base class for defining basic DataFrame configuration requirements.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        custom_index: list | None = None,
        target_column: str | None = None,
        column_name_to_drop_rows: str | None = None,
        value_in_row: str | float | int | None = None,
    ):

        self.dataframe = dataframe
        self.custom_index = list() if custom_index is None else custom_index
        self.target_column = target_column
        self.column_name_to_drop_rows = column_name_to_drop_rows
        self.value_in_row = value_in_row

    @abstractmethod
    def index_present(self) -> bool:
        """Validate the presence of an appropriate index"""

    @abstractmethod
    def target_column_present(self) -> bool:
        """Validate the presence of the target column"""

    @abstractmethod
    def explanatory_data(self) -> bool:
        """Validate the presence of explanatory variables"""

    @abstractmethod
    def is_data_numeric(self) -> bool:
        """Validate that the data is numeric"""

    @abstractmethod
    def data_correct_shape(self) -> bool:
        """Validate the shape of the DataFrame"""


class ValidateData(BaseDataClass):
    """
    Validation class to ensure data tidying was successful.
    """

    def index_present(self) -> bool:
        """
        Check if a custom index is used or if the default index is acceptable.
        """
        if self.custom_index:

            missing_columns = [
                col for col in self.custom_index if col not in self.dataframe.index.names
            ]
            if missing_columns:
                raise ValueError(f"Custom index columns missing: {missing_columns}")

            if isinstance(self.dataframe.index, pd.MultiIndex):
                index_names = self.dataframe.index.names
                if not all(name in self.custom_index for name in index_names):
                    raise ValueError("MultiIndex does not match custom index")

        return True

    def target_column_present(self) -> bool:
        """
        Verify the presence of the target column.
        """
        if self.target_column is None:
            raise ValueError("Target column not specified")

        if self.target_column not in self.dataframe.columns:
            raise ValueError(f"Target column '{self.target_column}' not present in data")

        return True

    def explanatory_data(self) -> bool:
        """
        Check for explanatory variables.
        """
        explanatory_columns = [
            col for col in self.dataframe.columns if col != self.target_column
        ]

        if len(explanatory_columns) == 0:
            raise ValueError("No explanatory data present")

        if len(explanatory_columns) < 2:
            warnings.warn(
                "Only one explanatory variable present. More variables are generally advised for machine learning modeling."
            )

        return True

    def is_data_numeric(self) -> bool:
        """
        Check if all columns (except target) are numeric.
        """
        non_target_columns = [
            col for col in self.dataframe.columns if col != self.target_column
        ]

        non_numeric_columns = [
            col
            for col in non_target_columns
            if self.dataframe[col].dtype not in ["int64", "float64"]
        ]

        if non_numeric_columns:
            raise ValueError(f"Non-numeric columns found: {non_numeric_columns}")

        return True

    def data_correct_shape(self) -> bool:
        """
        Validate DataFrame structure.
        """
        if self.dataframe.empty:
            raise ValueError("DataFrame is empty")

        if self.target_column not in self.dataframe.columns:
            raise ValueError(f'Target column "{self.target_column}" not found in DataFrame')

        explanatory_columns = [
            col for col in self.dataframe.columns if col != self.target_column
        ]

        if not explanatory_columns:
            raise ValueError("No explanatory variables found in DataFrame")

        return True

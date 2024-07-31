"""
Interface for predictors to implement.
"""
from abc import ABC, abstractmethod

import pandas as pd


class Predictor(ABC):
    """
    Interface for predictors to implement.
    Predictors must be able to be fit and predict on a DataFrame.
    It is up to the Predictor to keep track of the proper label to label the output DataFrame.
    """
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the model to the training data.
        :param X_train: DataFrame with input data:
            The input data consists of a DataFrame with context and actions.
            It is up to the model to decide which columns to use.
        :param y_train: pandas Series with target data.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a DataFrame with predictions for the input DataFrame.
        The Predictor model is expected to keep track of the label so that it can label the output
        DataFrame properly. Additionally, the output DataFrame must have the same index as the input DataFrame.
        :param context_actions_df: DataFrame with context and actions input data.
        :return: DataFrame with predictions
        """
        raise NotImplementedError

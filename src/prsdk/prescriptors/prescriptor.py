"""
Abstract prescriptor class to be implemented.
"""
from abc import ABC, abstractmethod

import pandas as pd


class Prescriptor(ABC):
    """
    Abstract class for prescriptors to allow us to experiment with different implementations.
    """
    def __init__(self, context: list[str], actions: list[str]):
        # We keep track of the context and actions to ensure that the prescriptor is compatible with the environment.
        self.context = context
        self.actions = actions

    @abstractmethod
    def prescribe(self, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes in a context dataframe and prescribes actions.
        Outputs a concatenation of the context and actions.
        :param context_df: A dataframe containing rows of context data.
        :return: A dataframe containing the context and the prescribed actions.
        """
        raise NotImplementedError

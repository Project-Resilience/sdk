"""
Interface prescriptor to be implemented.
"""
from abc import ABC, abstractmethod

import pandas as pd


# pylint: disable=too-few-public-methods
class Prescriptor(ABC):
    """
    Interface for prescriptors to implement.
    """
    @abstractmethod
    def prescribe(self, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes in a context dataframe and prescribes actions.
        Outputs a concatenation of the context and actions.
        :param context_df: A dataframe containing rows of context data.
        :return: A dataframe containing the context and the prescribed actions.
        """
        raise NotImplementedError
# pylint: enable=too-few-public-methods

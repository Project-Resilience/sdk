"""
Abstract prescriptor class to be implemented.
"""
from abc import ABC, abstractmethod

import pandas as pd

from data.cao_mapping import CAOMapping


# pylint: disable=too-few-public-methods
class Prescriptor(ABC):
    """
    Abstract class for prescriptors to allow us to experiment with different implementations.
    """
    def __init__(self, cao: CAOMapping):
        """
        We keep track of the context, actions, and outcomes in the CAO mapping to ensure the prescriptor is compatible
        with the project it's in.
        :param cao: CAOMapping object with context, actions, and outcomes.
        """
        self.cao = cao

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

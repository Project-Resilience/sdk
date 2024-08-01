"""
Interface for a parent selector to implement.
"""
from abc import ABC, abstractmethod

from prsdk.nsga2.candidate.candidate import Candidate


class ParentSelector(ABC):
    """
    Takes a list of sorted parents and selects n parents to mate.
    """
    @abstractmethod
    def select_parents(self, sorted_parents: list[Candidate], n: int) -> list[Candidate]:
        """
        Selects n parents to mate. Parents should be sorted in descending order.
        """
        raise NotImplementedError

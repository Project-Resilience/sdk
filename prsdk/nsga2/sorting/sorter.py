"""
Interface for a sorter to implement.
"""
from abc import ABC, abstractmethod

from prsdk.nsga2.candidate.candidate import Candidate


class Sorter(ABC):
    """
    Interface that handles the sorting of candidates after they are evaluated.
    """
    @abstractmethod
    def sort_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        """
        Sorts candidates based on some criteria. Returns the sorted list.
        :param candidates: List of candidates to sort.
        """
        raise NotImplementedError

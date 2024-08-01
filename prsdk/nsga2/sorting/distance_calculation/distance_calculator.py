"""
Interface for distance calculators to implement.
"""
from abc import ABC, abstractmethod

from prsdk.nsga2.candidate.candidate import Candidate


class DistanceCalculator(ABC):
    """
    Interface to calculate the distance between two candidates for NSGA-II sorting.
    """
    @abstractmethod
    def calculate_distance(self, front: list[Candidate]) -> None:
        """
        Calculates the distances of each candidate in the front. The distance is stored in the candidate.
        """
        raise NotImplementedError


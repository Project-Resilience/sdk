"""
Crossover interface to be implemented.
"""
from abc import ABC, abstractmethod

from prsdk.nsga2.candidate.candidate import Candidate


class Crossover(ABC):
    """
    Interface for crossover operations.
    """
    @abstractmethod
    def crossover(self, cand_id: str, parents: list[Candidate]) -> list[Candidate]:
        """
        Crosses over n parents to create m offspring.
        """
        raise NotImplementedError("crossover not implemented")

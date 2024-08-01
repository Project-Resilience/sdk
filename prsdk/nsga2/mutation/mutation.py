from abc import ABC, abstractmethod

from prsdk.nsga2.candidate.candidate import Candidate


class Mutation(ABC):
    """
    Abstract class handling mutation.
    """
    @abstractmethod
    def mutate(self, candidate: Candidate):
        """
        Mutate candidate.
        """
        raise NotImplementedError

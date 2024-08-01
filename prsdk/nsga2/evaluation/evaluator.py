"""
Evaluates candidates in order for them to be sorted.
"""
from abc import ABC, abstractmethod

from tqdm import tqdm

from prsdk.nsga2.candidate.candidate import Candidate


class Evaluator(ABC):
    """
    Abstract class for evaluating candidates.
    """
    @abstractmethod
    def evaluate_candidate(self, candidate: Candidate):
        """
        Evaluates a single candidate.
        """
        raise NotImplementedError("evaluate_candidate not implemented")

    def evaluate_candidates(self, candidates: list[Candidate], force=False):
        """
        Evaluates all candidates. Doesn't unnecessarily evaluate candidates that have already been evaluated unless the
        force flag is set.
        """
        for candidate in tqdm(candidates, leave=False):
            if len(candidate.metrics) == 0 or force:
                self.evaluate_candidate(candidate)

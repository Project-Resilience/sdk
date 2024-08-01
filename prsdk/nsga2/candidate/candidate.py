"""
Abstract class for Candidates to inherit from.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class Candidate(ABC):
    """
    Abstract class that represents a candidate in the evolutionary algorithm.
    Keeps track of candidate's ID and provides a method to record the state of the candidate.
    Also keeps track of the candidate's metrics for evaluation and sorting.
    """
    def __init__(self, cand_id: str, parents: list["Candidate"]):
        """
        :param cand_id: The ID of the candidate. Must be in the format "{generation}_{index}".
        """
        assert "_" in cand_id and len(cand_id.split("_")) == 2, "Candidate ID must be in the format generation_index"
        self.cand_id = cand_id
        self.parents = parents
        self.metrics = {}
        self.sorting_metrics = {}

    def record_state(self) -> dict:
        """
        Records the state of the candidate as a dictionary for logging purposes.
        """
        return {"cand_id": self.cand_id, "parents": self.parents, **self.sorting_metrics, **self.metrics}
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Saves the candidate to disk.
        """
        raise NotImplementedError("Save method must be implemented by subclass")

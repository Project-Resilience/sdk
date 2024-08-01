"""
Tournament selection implementation of ParentSelector.
"""
import numpy as np

from prsdk.nsga2.candidate.candidate import Candidate
from prsdk.nsga2.selection.parent_selector import ParentSelector


class TournamentSelector(ParentSelector):
    """
    Selects parents by doing tournament selection twice.
    """
    def __init__(self, remove_population_pct: float):
        self.remove_population_pct = remove_population_pct
        self.rng = np.random.default_rng()

    def tournament_selection(self, sorted_parents: list[Candidate]) -> Candidate:
        """
        Takes 2 random parents and picks the fittest one.
        """
        # Set cutoff to 1 if we are trying to remove more candidates than there are
        cutoff = max(int(len(sorted_parents) * (1 - self.remove_population_pct)), 1)
        top_parents = sorted_parents[:cutoff]
        return top_parents[min(self.rng.choice(len(top_parents), size=2, replace=True, shuffle=False))]

    def select_parents(self, sorted_parents: list[Candidate], n=2) -> list[Candidate]:
        """
        Selects n parents to mate.
        """
        return [self.tournament_selection(sorted_parents) for _ in range(n)]

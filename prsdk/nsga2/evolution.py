"""
PyTorch implementation of NSGA-II.
"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from prsdk.nsga2.candidate.candidate import Candidate
from prsdk.nsga2.crossover.crossover import Crossover
from prsdk.nsga2.evaluation.evaluator import Evaluator
from prsdk.nsga2.mutation.mutation import Mutation
from prsdk.nsga2.selection.parent_selector import ParentSelector
from prsdk.nsga2.sorting.sorter import Sorter


class Evolution:
    """
    Class handling the running of evolution.
    """
    def __init__(self,
                 save_path: Path,
                 evolution_params: dict,
                 parent_selector: ParentSelector,
                 crossover: Crossover,
                 mutator: Mutation,
                 sorter: Sorter,
                 evaluator: Evaluator):

        self.save_path = save_path

        # Evolution params
        self.pop_size = evolution_params["pop_size"]
        self.n_generations = evolution_params["n_generations"]
        self.n_elites = evolution_params["n_elites"]

        self.parent_selector = parent_selector
        self.crossover = crossover
        self.mutator = mutator
        self.sorter = sorter
        self.evaluator = evaluator

    def make_new_pop(self, candidates: list[Candidate], n: int, gen: int) -> list[Candidate]:
        """
        Creates new population of candidates.
        Doesn't remove any candidates from the previous generation if we're on the first generation.
        """
        children = []
        while len(children) < n:
            parents = self.parent_selector.select_parents(candidates)
            offspring = self.crossover.crossover(f"{gen}_{len(children)}", parents)
            offspring = self.mutator.mutate(offspring)
            children.extend(offspring)
        return children

    def record_gen_results(self, gen: int, candidates: list[Candidate]):
        """
        Logs results of generation to CSV. Saves candidates to disk
        """
        gen_results = [c.record_state() for c in candidates]
        gen_results_df = pd.DataFrame(gen_results)
        csv_path = self.save_path / f"{gen}.csv"
        gen_results_df.to_csv(csv_path, index=False)
        for c in candidates:
            if c.cand_id.startswith(str(gen)):
                c.save(self.save_path / str(gen) / f"{c.cand_id}.pt")

    def neuroevolution(self, initial_pop: list[Candidate] = None):
        """
        Main Neuroevolution Loop that performs NSGA-II.
        After initializing the first population randomly, goes through 3 steps in each generation:
        1. Evaluate candidates
        2. Select parents
        2a Log performance of parents
        3. Make new population from parents
        """
        print("Beginning evolution...")
        self.evaluator.evaluate_candidates(initial_pop)
        sorted_parents = self.sorter.sort_candidates(initial_pop)
        offspring = []
        for gen in tqdm(range(1, self.n_generations+1)):
            # Create offspring
            keep = min(self.n_elites, len(sorted_parents))
            offspring = self.make_new_pop(sorted_parents, self.pop_size-keep, gen)

            # Add elites to new generation
            # NOTE: The elites are also passed into the evaluation function. Make sure your
            # evaluator can handle this!
            new_parents = sorted_parents[:keep] + offspring
            self.evaluator.evaluate_candidates(new_parents)

            # Set rank and distance of parents
            sorted_parents = self.sorter.sort_candidates(new_parents)

            # Record the performance of the most successful candidates
            self.record_gen_results(gen, sorted_parents)

        return sorted_parents

def seed_first_gen(self):
    """
    Creates the first generation by taking seeds and creating an average of them.
    """
    candidates = []
    if self.seed_path:
        print("Seeding from ", self.seed_path)
        for seed in self.seed_path.iterdir():
            candidate = Candidate.from_seed(seed, self.model_params, self.actions, self.outcomes)
            candidates.append(candidate)

    print("Generating random seed generation")
    i = len(candidates)
    while i < self.pop_size:
        candidate = Candidate(f"0_{i}", [], self.model_params, self.actions, self.outcomes)
        candidates.append(candidate)
        i += 1

    self.evaluator.evaluate_candidates(candidates)
    candidates = self.sorter.sort_candidates(candidates)

    self.record_gen_results(0, candidates)
    return candidates

def generate_first_gen(self):
    """
    Randomly intializes the first generation of candidates.
    """
    candidates = []
    while len(candidates) < self.pop_size:
        candidates.append(Candidate(f"0_{len(candidates)}", [], self.model_params, self.actions, self.outcomes))
    return candidates
import torch

from prsdk.nsga2.candidate.candidate import Candidate
from prsdk.nsga2.mutation.mutation import Mutation


class UniformMutation(Mutation):
    """
    Uniformly mutates
    """
    def __init__(self, mutation_factor: float, mutation_rate: float):
        """
        :param mutation_factor: Std of the gaussian pct noise.
        :param mutation_rate: Probability of mutating a parameter.
        :param full: Mutate all parameters.
        :param same_mask: Use the same mask between runs of mutation.
        """
        self.mutation_factor = mutation_factor
        self.mutation_rate = mutation_rate

    def gaussian_pct_(self, param: torch.Tensor):
        """
        Mutate tensor with gaussian percentage in-place.
        mask must be passed into the function if we have one so that we can modify the model in-place.
        """
        mutate_mask = torch.rand(param.shape, device=param.device) < self.mutation_rate
        noise = torch.normal(0, self.mutation_factor, param[mutate_mask].shape, device=param.device, dtype=param.dtype)
        param[mutate_mask] += noise * param[mutate_mask]

    def mutate_(self, candidate: Candidate):
        """
        Mutate model with gaussian percentage in-place
        """
        for param in candidate.model.parameters():
            self.gaussian_pct_(param.data)

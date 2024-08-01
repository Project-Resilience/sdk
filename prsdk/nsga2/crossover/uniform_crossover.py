"""
Uniform crossover implementation.
"""
import torch

from prsdk.nsga2.candidate.torch_candidate import TorchCandidate
from prsdk.nsga2.crossover.crossover import Crossover


class UniformCrossover(Crossover):
    """
    Crosses over 2 parents.
    We do not keep track of what's in the models and assume they are loaded correct with the parents.
    """
    def crossover(self, cand_id: str, parents: list[TorchCandidate]) -> list[TorchCandidate]:
        child = TorchCandidate(cand_id=cand_id,
                               parents=[parents[0].cand_id, parents[1].cand_id],
                               model_params=parents[0].model_params)
        with torch.no_grad():
            child.model.load_state_dict(parents[0].model.state_dict())
            for param, param2 in zip(child.model.parameters(), parents[1].model.parameters()):
                mask = torch.rand(param.shape, device=param.device) < 0.5
                param.data[mask] = param2.data[mask]
        return [child]

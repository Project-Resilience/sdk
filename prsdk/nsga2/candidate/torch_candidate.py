"""
Sample implementation of a Candidate using a PyTorch neural network.
"""
from pathlib import Path

import torch

from prsdk.nsga2.candidate.candidate import Candidate


class TorchCandidate(Candidate):
    """
    Candidate implementation using a simple PyTorch neural network.
    NOTE: Any scaling or pandas happens outside the candidate in the Evaluator, or later in a Prescriptor wrapper around
    the Candidate. This is to speed things up by avoiding repeat computations such as data scaling.
    """
    def __init__(self, cand_id: str, parents: list["TorchCandidate"], model_params: dict):
        super().__init__(cand_id=cand_id, parents=parents)
        self.model = torch.nn.Sequential([
            torch.nn.Linear(model_params["in_size"], model_params["hidden_size"]),
            torch.nn.Tanh(),
            torch.nn.Linear(model_params["hidden_size"], model_params["out_size"])
        ])
        self.model.eval()
        # Store our model params for crossover later.
        self.model_params = model_params

    def random_init(self):
        """
        Performs orthogonal initialization to create a diverse randomly generated population.
        """
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.01)

    @classmethod
    def from_seed(cls, seed_path: Path, cand_id: str, model_params: dict) -> "Candidate":
        """
        Load candidate from seed path.
        """
        candidate = cls(cand_id=cand_id, parents=[], model_params=model_params)
        candidate.model.load_state_dict(torch.load(seed_path, map_location="cpu"))
        return candidate

    def save(self, path: Path) -> None:
        """
        Saves PyTorch state dict to disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def prescribe(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """
        Runs the context tensor through the model.
        """
        return self.model(context_tensor)

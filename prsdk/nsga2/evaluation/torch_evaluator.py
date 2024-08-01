import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from prsdk.data.torch_data import TorchDataset
from prsdk.nsga2.candidate.torch_candidate import TorchCandidate
from prsdk.nsga2.evaluation.evaluator import Evaluator
from prsdk.predictors.neural_network.neural_net_predictor import NeuralNetPredictor


class TorchEvaluator(Evaluator):

    def __init__(self, eval_df: pd.DataFrame, outcomes: dict[str, bool], predictor: NeuralNetPredictor):
        self.scaler = StandardScaler()
        encoded_df = self.scaler.fit_transform(eval_df)
        encoded_context_df = encoded_df.drop(columns=list(outcomes.keys()))
        dataset = TorchDataset(encoded_context_df.values, np.zeros(len(encoded_context_df)))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        self.predictor = predictor

    def evaluate_candidate(self, candidate: TorchCandidate) -> dict[str, float]:
        with torch.no_grad():
            metrics = {}
            outcomes_list = []
            for context_tensor, _ in self.dataloader:
                actions_tensor = candidate.prescribe(context_tensor)
                context_actions_tensor = torch.cat((context_tensor, actions_tensor), dim=1)
                outcomes = self.predictor.forward(context_actions_tensor)
                outcomes_list.append(outcomes)
            outcomes_tensor = torch.stack(outcomes_list)

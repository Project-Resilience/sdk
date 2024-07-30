"""
Unit tests for the NeuralNetPredictor class.
"""
import unittest

import pandas as pd

from data.cao_mapping import CAOMapping
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor


class TestNeuralNet(unittest.TestCase):
    """
    Specifically tests the neural net predictor
    """
    def setUp(self):
        self.cao = CAOMapping(["a", "b"], ["c"], ["label"])

    def test_single_input(self):
        """
        Tests the neural net with a single input.
        """
        predictor = NeuralNetPredictor(self.cao, {"hidden_sizes": [4], "epochs": 1, "batch_size": 1, "device": "cpu"})

        train_data = pd.DataFrame({"a": [1], "b": [2], "c": [3], "label": [4]})
        test_data = pd.DataFrame({"a": [4], "b": [5], "c": [6]})

        predictor.fit(train_data[['a', 'b', 'c']], train_data['label'])
        out = predictor.predict(test_data)
        self.assertEqual(out.shape, (1, 1))

    def test_multi_input(self):
        """
        Tests the neural net with multiple inputs.
        """
        predictor = NeuralNetPredictor(self.cao, {"hidden_sizes": [4], "epochs": 1, "batch_size": 1, "device": "cpu"})

        train_data = pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4], "label": [4, 5]})
        test_data = pd.DataFrame({"a": [4, 5], "b": [5, 6], "c": [6, 7]})

        predictor.fit(train_data[['a', 'b', 'c']], train_data['label'])
        out = predictor.predict(test_data)
        self.assertEqual(out.shape, (2, 1))

    def test_batched_input(self):
        """
        Tests the neural network with batched inputs.
        """
        predictor = NeuralNetPredictor(self.cao, {"hidden_sizes": [4], "epochs": 1, "batch_size": 2, "device": "cpu"})

        train_data = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5], "label": [4, 5, 6]})
        test_data = pd.DataFrame({"a": [4, 5], "b": [5, 6], "c": [6, 7]})

        predictor.fit(train_data[['a', 'b', 'c']], train_data['label'])
        out = predictor.predict(test_data)
        self.assertEqual(out.shape, (2, 1))

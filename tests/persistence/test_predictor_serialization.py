"""
Unit tests for the predictors.
"""
import unittest
import shutil
from pathlib import Path

import pandas as pd

from prsdk.persistence.serializers.neural_network_serializer import NeuralNetSerializer
from prsdk.persistence.serializers.sklearn_serializer import SKLearnSerializer
from prsdk.predictors.neural_network.neural_net_predictor import NeuralNetPredictor
from prsdk.predictors.sklearn_predictors.linear_regression_predictor import LinearRegressionPredictor
from prsdk.predictors.sklearn_predictors.random_forest_predictor import RandomForestPredictor


class TestPredictorSerialization(unittest.TestCase):
    """
    Tests the 3 base predictor implementations' saving and loading behavior.
    """
    def setUp(self):
        """
        We set the models up like this so that in test_loaded_same we can instantiate
        2 models with the same parameters, load one from the other's save, and check if
        their predictions are the same.
        """
        self.models = [
            NeuralNetPredictor,
            LinearRegressionPredictor,
            RandomForestPredictor
        ]
        self.serializers = [
            NeuralNetSerializer(),
            SKLearnSerializer(),
            SKLearnSerializer()
        ]
        self.configs = [
            {'hidden_sizes': [4], 'epochs': 1, 'batch_size': 1, 'device': 'cpu'},
            {'n_jobs': -1},
            {'n_jobs': -1, "n_estimators": 10, "max_depth": 2}
        ]
        self.dummy_data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 4], "c": [7, 8, 9, 4]})
        self.dummy_target = pd.Series([1, 2, 3, 4], name="label")
        self.temp_path = Path("tests/temp")

    def test_save_file_names(self):
        """
        Checks to make sure the model's save method creates the correct files.
        """
        save_file_names = [
            ["model.pt", "config.json", "scaler.joblib"],
            ["model.joblib", "config.json"],
            ["model.joblib", "config.json"]
        ]
        for model, serializer, config, test_names in zip(self.models, self.serializers, self.configs, save_file_names):
            with self.subTest(model=model):
                predictor = model(config)
                predictor.fit(self.dummy_data, self.dummy_target)
                serializer.save(predictor, self.temp_path)
                files = [f.name for f in self.temp_path.glob("**/*") if f.is_file()]
                self.assertEqual(set(files), set(test_names))
                shutil.rmtree(self.temp_path)
                self.assertFalse(self.temp_path.exists())

    def test_loaded_same(self):
        """
        Makes sure a predictor's predictions are consistent before and after saving/loading.
        Fits a predictor then saves and loads it, then checks if the predictions are the same.
        """
        for model, serializer, config in zip(self.models, self.serializers, self.configs):
            with self.subTest(model=model):
                predictor = model(config)
                predictor.fit(self.dummy_data.iloc[:2], self.dummy_target.iloc[:2])
                output = predictor.predict(self.dummy_data.iloc[2:])
                serializer.save(predictor, self.temp_path)

                loaded = serializer.load(self.temp_path)
                loaded_output = loaded.predict(self.dummy_data.iloc[2:])

                self.assertTrue((output == loaded_output).all().all())
                shutil.rmtree(self.temp_path)
                self.assertFalse(self.temp_path.exists())

    def tearDown(self):
        """
        Removes the temp directory if it exists.
        """
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)

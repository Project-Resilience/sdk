"""
Unit tests for the HuggingFace persistor.
"""
from pathlib import Path
import shutil
import unittest

import numpy as np
import pandas as pd

from persistence.persistors.hf_persistor import HuggingFacePersistor
from persistence.serializers.neural_network_serializer import NeuralNetSerializer
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor


class TestHuggingFacePersistence(unittest.TestCase):
    """
    Tests the HuggingFace Persistor. We can't test the actual upload but we can test the download with an
    arbitrary model from HuggingFace.
    """
    def setUp(self):
        self.temp_dir = Path("tests/temp")

    def test_download_model(self):
        """
        Tests downloading a model from HuggingFace.
        """
        url = "danyoung/eluc-global-nn"
        serializer = NeuralNetSerializer()
        persistor = HuggingFacePersistor(serializer)

        model = persistor.from_pretrained(url, local_dir=str(self.temp_dir / url.replace("/", "--")))
        self.assertTrue(isinstance(model, NeuralNetPredictor))
        self.assertTrue((self.temp_dir / url.replace("/", "--") / "config.json").exists())
        self.assertTrue((self.temp_dir / url.replace("/", "--") / "model.pt").exists())
        self.assertTrue((self.temp_dir / url.replace("/", "--") / "scaler.joblib").exists())

    def test_predict_model(self):
        """
        Tests that you can predict with a downloaded model.
        """
        url = "danyoung/eluc-global-nn"
        serializer = NeuralNetSerializer()
        persistor = HuggingFacePersistor(serializer)

        model = persistor.from_pretrained(url, local_dir=str(self.temp_dir / url.replace("/", "--")))
        test_data = pd.DataFrame({cont: np.random.rand(5) for cont in model.features})
        out = model.predict(test_data)
        self.assertEqual(out.shape, (5, 1))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

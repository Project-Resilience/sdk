"""
Serializer for the SKLearnPredictor class.
"""
import json
from pathlib import Path

import joblib

from data.cao_mapping import CAOMapping
from persistence.serializers.serializer import Serializer
from predictors.sklearn_predictors.sklearn_predictor import SKLearnPredictor


class SKLearnSerializer(Serializer):
    """
    Serializer for the SKLearnPredictor.
    Uses joblib to save the model and json to save the config used to load it.
    """
    def save(self, model: SKLearnPredictor, path: Path):
        """
        Saves saves model and features into format for loading.
        Generates path to folder if it does not exist.
        :param path: path to folder to save model files.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Add CAO to the config
        config = dict(model.config.items())
        cao_dict = {"context": model.cao.context, "actions": model.cao.actions, "outcomes": model.cao.outcomes}
        config.update(cao_dict)

        with open(path / "config.json", "w", encoding="utf-8") as file:
            json.dump(config, file)
        joblib.dump(model.model, path / "model.joblib")

    def load(self, path: Path) -> "SKLearnPredictor":
        """
        Loads saved model and config from a local folder.
        :param path: path to folder to load model files from.
        """
        load_path = Path(path)
        if not load_path.exists() or not load_path.is_dir():
            raise FileNotFoundError(f"Path {path} does not exist.")
        if not (load_path / "config.json").exists() or not (load_path / "model.joblib").exists():
            raise FileNotFoundError("Model files not found in path.")

        # Extract CAO from config
        with open(load_path / "config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
        cao = CAOMapping(config.pop("context"), config.pop("actions"), config.pop("outcomes"))

        model = joblib.load(load_path / "model.joblib")
        sklearn_predictor = SKLearnPredictor(cao, model, config)
        return sklearn_predictor

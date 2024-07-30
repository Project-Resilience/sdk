"""
Persistor abstract class. Wraps a serializer and provides an interface for persisting models
(ex to HuggingFace) and loading models from a persistence location.
"""
from pathlib import Path

from abc import ABC, abstractmethod

from prsdk.persistence.serializers.serializer import Serializer


class Persistor(ABC):
    """
    Abstract class for persistors to inherit from.
    Wraps around a serializer to cache the persisted models onto disk before loading them.
    """
    def __init__(self, serializer: Serializer):
        self.serializer = serializer

    @abstractmethod
    def persist(self, model, model_path: Path, repo_id: str, **persistence_args):
        """
        Serializes a model using the serializer, then uploads the model to a persistence location.
        :param model: The python object model to persist.
        :param model_path: The path on disk to save the model to before persisting it.
        :param repo_id: The ID used to point to the model in whatever method we use to persist it.
        :param persistence_args: Additional arguments to pass to the persistence method.
        """
        raise NotImplementedError("Persisting not implemented")

    @abstractmethod
    def from_pretrained(self, path_or_url: str, **persistence_args):
        """
        Loads a model from where it was persisted from.
        :param path_or_url: The path or URL to load the model from.
        :param persistence_args: Additional arguments to pass to the loading method.
        """
        raise NotImplementedError("Loading not implemented")

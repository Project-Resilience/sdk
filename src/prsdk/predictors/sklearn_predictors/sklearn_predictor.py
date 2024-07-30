"""
Abstract SKLearn predictor.
Since the SKLearn library is standardized we can easily make more.
"""
from abc import ABC

import pandas as pd

from data.cao_mapping import CAOMapping
from predictors.predictor import Predictor


class SKLearnPredictor(Predictor, ABC):
    """
    Simple abstract class for sklearn predictors.
    Keeps track of features fit on and label to predict.
    """
    def __init__(self, cao: CAOMapping, model, model_config: dict):
        """
        Model config contains the following:
        features: list of features to use for prediction (optional, defaults to all features)
        label: name of the label to predict (optional, defaults to passed label during fit)
        Any other parameters are passed to the model.
        """
        super().__init__(cao)
        self.config = model_config
        self.model = model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits SKLearn model with standard sklearn fit method.
        If we passed in features, use those. Otherwise use all columns.
        :param X_train: DataFrame with input data
        :param y_train: series with target data
        """
        if "features" in self.config:
            X_train = X_train[self.config["features"]]
        else:
            self.config["features"] = list(X_train.columns)
        self.config["label"] = y_train.name
        self.model.fit(X_train, y_train)

    def predict(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard sklearn predict method.
        Makes sure to use the same features as were used in fit.
        Ensures index and label are properly applied to the output.
        :param context_actions_df: DataFrame with input data
        :return: properly labeled DataFrame with predictions and matching index.
        """
        context_actions_df = context_actions_df[self.config["features"]]
        y_pred = self.model.predict(context_actions_df)
        return pd.DataFrame(y_pred, index=context_actions_df.index, columns=[self.config["label"]])

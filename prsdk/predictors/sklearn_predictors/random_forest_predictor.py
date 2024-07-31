"""
Implementation of SKLearnPredictor as a RandomForestRegressor.
"""
from sklearn.ensemble import RandomForestRegressor

from prsdk.predictors.sklearn_predictors.sklearn_predictor import SKLearnPredictor


class RandomForestPredictor(SKLearnPredictor):
    """
    Simple random forest predictor.
    See SKLearnPredictor for more details.
    """
    def __init__(self, model_config: dict):
        """
        :param model_config: Configuration to pass into the SKLearn constructor. Also contains the keys "features" and
            "label" to keep track of the features and label to predict.
        """
        rf_config = {key: value for key, value in model_config.items() if key not in ["features", "label"]}
        model = RandomForestRegressor(**rf_config)
        super().__init__(model, model_config)

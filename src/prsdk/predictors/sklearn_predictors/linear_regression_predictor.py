"""
Implementation of SKLearnPredictor as a LinearRegressor.
"""
from sklearn.linear_model import LinearRegression

from predictors.sklearn_predictors.sklearn_predictor import SKLearnPredictor


class LinearRegressionPredictor(SKLearnPredictor):
    """
    Simple linear regression predictor.
    See SKLearnPredictor for more details.
    """
    def __init__(self, model_config: dict):
        """
        :param model_config: Configuration to pass into the SKLearn constructor. Also contains the keys "features" and
            "label" to keep track of the features and label to predict.
        """
        if not model_config:
            model_config = {}
        lr_config = {key: value for key, value in model_config.items() if key not in ["features", "label"]}
        model = LinearRegression(**lr_config)
        super().__init__(model, model_config)

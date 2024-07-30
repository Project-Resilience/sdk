"""
Implementation of predictor.py using a simple feed-forward NeuralNetwork
implemented in PyTorch.
"""
import copy
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.cao_mapping import CAOMapping
from data.torch_data import TorchDataset
from predictors.predictor import Predictor
from predictors.neural_network.torch_neural_net import TorchNeuralNet


# pylint: disable=too-many-instance-attributes
class NeuralNetPredictor(Predictor):
    """
    Simple feed-forward neural network predictor implemented in PyTorch.
    Has the option to use wide and deep, concatenating the input to the output of the hidden layers
    in order to take advantage of the linear relationship in the data.
    Data is automatically standardized and the scaler is saved with the model.
    TODO: We want to be able to have custom scaling in the future.
    """
    def __init__(self, cao: CAOMapping, model_config: dict):
        """
        :param context: list of context features.
        :param actions: list of action features.
        :param outcomes: list of outcomes to predict.
        :param model_config: dictionary of model configuration parameters.
            Model config should contain the following:
            features: list of features to use in the model (optional, defaults to all context + actions)
            label: name of the label column (optional, defaults to passed label in fit)
            hidden_sizes: list of hidden layer sizes (defaults to single layer of size 4096)
            linear_skip: whether to concatenate input to hidden layer output (defaults to True)
            dropout: dropout probability (defaults to 0)
            device: device to run the model on (defaults to "cpu")
            epochs: number of epochs to train for (defaults to 3)
            batch_size: batch size for training (defaults to 2048)
            optim_params: dictionary of parameters to pass to the optimizer (defaults to PyTorch default)
            train_pct: percentage of training data to use (defaults to 1)
            step_lr_params: dictionary of parameters to pass to the step learning rate scheduler (defaults to 1, 0.1)
        """
        super().__init__(cao)
        self.features = model_config.get("features", None)
        self.label = model_config.get("label", None)

        self.hidden_sizes = model_config.get("hidden_sizes", [4096])
        self.linear_skip = model_config.get("linear_skip", True)
        self.dropout = model_config.get("dropout", 0)
        self.device = model_config.get("device", "cpu")
        self.epochs = model_config.get("epochs", 3)
        self.batch_size = model_config.get("batch_size", 2048)
        self.optim_params = model_config.get("optim_params", {})
        self.train_pct = model_config.get("train_pct", 1)
        self.step_lr_params = model_config.get("step_lr_params", {"step_size": 1, "gamma": 0.1})

        self.model = None
        self.scaler = StandardScaler()

    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val=None, y_val=None,
            X_test=None, y_test=None,
            log_path=None, verbose=False) -> dict:
        """
        Fits neural network to given data using predefined parameters and hyperparameters.
        If no features were specified we use all the columns in X_train.
        We scale based on the training data and apply it to validation and test data.
        AdamW optimizer is used with L1 loss.
        TODO: We want to be able to customize the loss function in the future.
        :param X_train: training data, may be unscaled and have excess features.
        :param y_train: training labels.
        :param X_val: validation data, may be unscaled and have excess features.
        :param y_val: validation labels.
        :param X_test: test data, may be unscaled and have excess features.
        :param y_test: test labels.
        :param log_path: path to log training data to tensorboard.
        :param verbose: whether to print progress bars.
        :return: dictionary of results from training containing time taken, best epoch, best loss,
        and test loss if applicable.
        """
        if not self.features:
            self.features = X_train.columns.tolist()
        self.label = y_train.name

        self.model = TorchNeuralNet(len(self.features), self.hidden_sizes, self.linear_skip, self.dropout)
        self.model.to(self.device)
        self.model.train()

        start = time.time()

        # Set up train set
        X_train = self.scaler.fit_transform(X_train[self.features])
        y_train = y_train.values
        train_ds = TorchDataset(X_train, y_train)
        sampler = torch.utils.data.RandomSampler(train_ds, num_samples=int(len(train_ds) * self.train_pct))
        train_dl = DataLoader(train_ds, self.batch_size, sampler=sampler)

        # If we pass in a validation set, use them
        if X_val is not None and y_val is not None:
            X_val = self.scaler.transform(X_val[self.features])
            y_val = y_val.values
            val_ds = TorchDataset(X_val, y_val)
            val_dl = DataLoader(val_ds, self.batch_size, shuffle=False)

        # Optimization parameters
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.optim_params)
        loss_fn = torch.nn.L1Loss()
        if self.step_lr_params:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.step_lr_params)

        if log_path:
            writer = SummaryWriter(log_path)

        # Keeping track of best performance for validation
        result_dict = {}
        best_model = None
        best_loss = np.inf
        end = 0

        step = 0
        for epoch in range(self.epochs):
            self.model.train()
            # Standard training loop
            train_iter = tqdm(train_dl) if verbose else train_dl
            for X, y in train_iter:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(X)
                loss = loss_fn(out.squeeze(), y.squeeze())
                if log_path:
                    writer.add_scalar("loss", loss.item(), step)
                step += 1
                loss.backward()
                optimizer.step()

            # LR Decay
            if self.step_lr_params:
                scheduler.step()

            # Evaluate epoch
            if X_val is not None and y_val is not None:
                total = 0
                self.model.eval()
                with torch.no_grad():
                    for X, y in tqdm(val_dl):
                        X, y = X.to(self.device), y.to(self.device)
                        out = self.model(X)
                        loss = loss_fn(out.squeeze(), y.squeeze())
                        total += loss.item() * y.shape[0]

                if log_path:
                    writer.add_scalar("val_loss", total / len(val_ds), step)

                if total < best_loss:
                    best_model = copy.deepcopy(self.model.state_dict())
                    best_loss = total
                    end = time.time()
                    result_dict["best_epoch"] = epoch
                    result_dict["best_loss"] = total / len(val_ds)
                    result_dict["time"] = end - start

                print(f"epoch {epoch} mae {total / len(val_ds)}")

        if best_model:
            self.model.load_state_dict(best_model)
        else:
            end = time.time()
            result_dict["time"] = end - start

        # If we provide a test dataset
        if X_test is not None and y_test is not None:
            y_pred = self.predict(X_test)
            y_true = y_test.values
            mae = np.mean(np.abs(y_pred - y_true))
            result_dict["test_loss"] = mae

        return result_dict
    # pylint: enable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements

    def predict(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates prediction from model for given test data.
        :param context_actions_df: test data to predict on.
        :return: DataFrame of predictions properly labeled and indexed.
        """
        X_test_scaled = self.scaler.transform(context_actions_df[self.features])
        test_ds = TorchDataset(X_test_scaled, np.zeros(len(X_test_scaled)))
        test_dl = DataLoader(test_ds, self.batch_size, shuffle=False)
        pred_list = []
        with torch.no_grad():
            self.model.eval()
            for X, _ in test_dl:
                X = X.to(self.device)
                pred_list.append(self.model(X))

        # Flatten into a single numpy array if we have multiple batches
        if len(pred_list) > 1:
            y_pred = torch.concatenate(pred_list, dim=0).cpu().numpy()
        else:
            y_pred = pred_list[0].cpu().numpy()
        return pd.DataFrame(y_pred, index=context_actions_df.index, columns=[self.label])

    def set_device(self, device: str):
        """
        Sets the device to run the model on.
        """
        self.device = device
        if self.model:
            self.model.to(device)

import torch
"""Neural Network model for the CAM-EM."""

import netCDF4 as nc
import numpy as np
from torch import nn
from torch.utils.data import Dataset


# Required for feeding the data iinto NN.
class myDataset(Dataset):
    def __init__(self, X, Y):
        """
        Parameters:
            X (tensor): Input data.
            Y (tensor): Output data.
        """
        self.features = torch.tensor(X, dtype=torch.float64)
        self.labels = torch.tensor(Y, dtype=torch.float64)

    def __len__(self):
        """Function that is called when you call len(dataloader)"""
        return len(self.features.T)

    def __getitem__(self, idx):
        """Function that is called when you call dataloader"""
        feature = self.features[:, idx]
        label = self.labels[:, idx]

        return feature, label


# The NN model.
class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

class FullyConnected(nn.Module):
    """
    Fully connected neural network model.

    Attributes
    ----------
    linear_stack : nn.Sequential
        Sequential container of linear layers and activation functions.
    """
    def __init__(self, ilev=93, hidden_layers=8, hidden_size=500):
        super(FullyConnected, self).__init__()
        layers = []
        input_size = 8 * ilev + 4  
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size, dtype=torch.float64))
            layers.append(nn.SiLU())
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, 2 * ilev, dtype=torch.float64))
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, X):
        """
        Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.linear_stack(X)

class EarlyStopper:
    """
    Early stopping utility to stop training when validation loss doesn't improve.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait before stopping (default is 1).
    min_delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement (default is 0).

    Attributes
    ----------
    patience : int
        Number of epochs to wait before stopping.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    counter : int
        Counter for the number of epochs without improvement.
    min_validation_loss : float
        Minimum validation loss recorded.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, model=None):
        """
        Check if training should be stopped early.

        Parameters
        ----------
        validation_loss : float
            Current validation loss.
        model : nn.Module, optional
            Model to save if validation loss improves (default is None).

        Returns
        -------
        bool
            True if training should be stopped, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if model is not None:
                torch.save(model.state_dict(), 'conv_torch.pth')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

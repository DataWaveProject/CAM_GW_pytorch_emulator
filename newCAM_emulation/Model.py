"""Neural Network model for the CAM-EM."""

import netCDF4 as nc
import numpy as np
import scipy.stats as st
import torch
import xarray as xr
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Dataset


# Required for feeding the data iinto NN.
class myDataset(Dataset):
    """
    Dataset class for loading features and labels.

    Args:
        X (numpy.ndarray): Input features.
        Y (numpy.ndarray): Corresponding labels.
    """

    def __init__(self, X, Y):
        """Create an instance of myDataset class."""
        self.features = torch.tensor(X, dtype=torch.float64)
        self.labels = torch.tensor(Y, dtype=torch.float64)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.features.T)

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
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

    The model consists of multiple fully connected layers with SiLU activation function.

    Attributes
    ----------
        linear_stack (torch.nn.Sequential): Sequential container for layers.
    """

    def __init__(self, ilev, mean, std):
        """Create an instance of FullyConnected NN model."""
        super(FullyConnected, self).__init__()
        self.normalization = NormalizationLayer(mean, std)
        self.ilev = ilev

        layers = []
        layers.append(nn.Linear(8 * ilev + 4, 500))
        layers.append(nn.SiLU())

        num_layers = 10  # Example: Change this to the desired number of hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(500, 500))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(500, 2 * ilev))
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Output tensor.
        """
        x = self.normalization(x)
        return self.linear_stack(x)

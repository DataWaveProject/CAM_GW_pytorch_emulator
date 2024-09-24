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
class FullyConnected(nn.Module):
    """
    Fully connected neural network model.

    The model consists of multiple fully connected layers with SiLU activation function.

    Attributes
    ----------
        linear_stack (torch.nn.Sequential): Sequential container for layers.
    """

    def __init__(self):
        """Create an instance of FullyConnected NN model."""
        super(FullyConnected, self).__init__()
        ilev = 93

        self.linear_stack = nn.Sequential(
            nn.Linear(8 * ilev + 4, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 2 * ilev, dtype=torch.float64),
        )

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Output tensor.
        """
        return self.linear_stack(X)


# training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Training loop.

    Args:
        dataloader (DataLoader): DataLoader for training data.
        model (nn.Module): Neural network model.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.

    Returns
    -------
        float: Average training loss.
    """
    size = len(dataloader.dataset)
    avg_loss = 0
    for batch, (X, Y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            avg_loss += loss.item()

    avg_loss /= len(dataloader)

    return avg_loss


# validating loop
def val_loop(dataloader, model, loss_fn):
    """
    Run the validation loop.

    Args:
        dataloader (DataLoader): DataLoader for validation data.
        model (nn.Module): Neural network model.
        loss_fn (torch.nn.Module): Loss function.

    Returns
    -------
        float: Average validation loss.
    """
    avg_loss = 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, Y)
            avg_loss += loss.item()

    avg_loss /= len(dataloader)

    return avg_loss

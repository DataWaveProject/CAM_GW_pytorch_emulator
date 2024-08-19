"""Neural Network model for the CAM-EM."""

import numpy as np
import torch
from torch import nn

# ruff: noqa: PLR0913

class FullyConnected(nn.Module):
    """
    Fully connected neural network model.

    Attributes
    ----------
    linear_stack : nn.Sequential
        Sequential container of linear layers and activation functions.
    """

    def __init__(
        self, ilev=93, in_ver=8, in_nover=4, out_ver=2,
        hidden_layers=8, hidden_size=500
    ):
        super(FullyConnected, self).__init__()
        self.ilev = ilev
        self.in_ver = in_ver
        self.in_nover = in_nover
        self.out_ver = out_ver
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        layers = []

        input_size = in_ver * ilev + in_nover

        # The following for loop provides the sequential layer by layer flow
        # of data in the model as the layers used in our model are identical.
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size, dtype=torch.float64))
            layers.append(nn.SiLU())
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, out_ver * ilev, dtype=torch.float64))
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
        Minimum change in the loss to qualify as an improvement (default is 0).

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
            # if model is not None:
            #     # torch.save(model.state_dict(), 'conv_torch.pth')
            #     torch.save(model.state_dict(), 'trained_models/weights_conv')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

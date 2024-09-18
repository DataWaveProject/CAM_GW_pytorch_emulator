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
        self, ilev=93, in_ver=8, in_nover=4, out_ver=2, hidden_layers=8, hidden_size=500
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

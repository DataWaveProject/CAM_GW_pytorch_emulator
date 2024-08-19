"""Training script for the neural network."""

import torch
from torch import nn

# ruff: noqa: PLR0913


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Training loop for a single epoch.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the training data.
    model : nn.Module
        Neural network model.
    loss_fn : callable
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for training.

    Returns
    -------
    float
        Average training loss.
    """
    avg_loss = 0
    for batch, (X, Y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss /= len(dataloader)
    return avg_loss


def val_loop(dataloader, model, loss_fn):
    """
        Validate loop for a single epoch.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the validation data.
    model : nn.Module
        Neural network model.
    loss_fn : callable
        Loss function.

    Returns
    -------
    float
        Average validation loss.
    """
    avg_loss = sum(loss_fn(model(X), Y).item() for X, Y in dataloader) / len(dataloader)
    return avg_loss


def train_with_early_stopping(
    train_dataloader,
    val_dataloader,
    model,
    optimizer,
    criterion,
    early_stopper,
    epochs=100,
):
    """
    Train the model with early stopping.

    Parameters
    ----------
    train_dataloader : torch.utils.data.DataLoader
        DataLoader for the training data.
    val_dataloader : torch.utils.data.DataLoader
        DataLoader for the validation data.
    model : nn.Module
        Neural network model.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    criterion : callable
        Loss function.
    early_stopper : EarlyStopper
        Early stopping utility.
    epochs : int, optional
        Number of epochs to train (default is 100).

    Returns
    -------
    tuple of list of float
        Training losses and validation losses for each epoch.
    """
    train_losses = []
    val_losses = [0]
    for epoch in range(epochs):
        if epoch % 2 == 0:
            print(f"Epoch {epoch + 1}\n-------------------------------")
            print(val_losses[-1])
            print("counter=" + str(early_stopper.counter))
        train_loss = train_loop(train_dataloader, model, criterion, optimizer)
        train_losses.append(train_loss)
        val_loss = val_loop(val_dataloader, model, criterion)
        val_losses.append(val_loss)
        if early_stopper.early_stop(val_loss, model):
            # print("BREAK!")
            break
    return train_losses, val_losses

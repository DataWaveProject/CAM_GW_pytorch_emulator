"""Training script for the neural network."""

import torch
from torch import nn

# ruff: noqa: PLR0913

# class EarlyStopper:
#     """
#     Early stopping utility to stop training when validation loss doesn't improve.

#     Parameters
#     ----------
#     patience : int, optional
#         Number of epochs to wait before stopping (default is 1).
#     min_delta : float, optional
#         Minimum change in the loss to qualify as an improvement (default is 0).

#     Attributes
#     ----------
#     patience : int
#         Number of epochs to wait before stopping.
#     min_delta : float
#         Minimum change in the monitored quantity to qualify as an improvement.
#     counter : int
#         Counter for the number of epochs without improvement.
#     min_validation_loss : float
#         Minimum validation loss recorded.
#     """

#     def __init__(self, patience=1, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = np.inf

#     def early_stop(self, validation_loss, model=None):
#         """
#         Check if training should be stopped early.

#         Parameters
#         ----------
#         validation_loss : float
#             Current validation loss.
#         model : nn.Module, optional
#             Model to save if validation loss improves (default is None).

#         Returns
#         -------
#         bool
#             True if training should be stopped, False otherwise.
#         """
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#             # if model is not None:
#             #     # torch.save(model.state_dict(), 'conv_torch.pth')
#             #     torch.save(model.state_dict(), 'trained_models/weights_conv')
#         elif validation_loss > (self.min_validation_loss + self.min_delta):
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False

def early_stopping(self, validation_loss, patience=1, min_delta=0, model=None):
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

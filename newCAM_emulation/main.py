"""Script to load data and train the neural network."""

import os

import numpy as np
import torch
from loaddata import (
    MyDataset,
    data_loader,
    load_mean_std,
    load_variables,
    normalize_data,
)
from Model import FullyConnected
from torch import nn
from torch.utils.data import DataLoader
from train import early_stopping, train_with_early_stopping

# File paths and parameters
directory_path = "Demodata"
file_path_mean = "Demodata/mean_demo_sub.npz"
file_path_std = "Demodata/std_demo_sub.npz"
trained_model_path = "trained_models/weights_conv"

# variable information
features = [
    "U",
    "V",
    "T",
    "DSE",
    "NM",
    "NETDT",
    "Z3",
    "RHOI",
    "PS",
    "lat",
    "lon",
    "UTGWSPEC",
    "VTGWSPEC",
]
ilev = 93
in_ver = 8
in_nover = 4
out_ver = 2

# Load and preprocess data
variable_data = load_variables(directory_path, features, 1, 5)
mean_dict, std_dict = load_mean_std(file_path_mean, file_path_std, features)
normalized_data = normalize_data(variable_data, mean_dict, std_dict)
xtrain, ytrain = data_loader(
    features,
    normalized_data,
    ilev=ilev,
    in_ver=in_ver,
    in_nover=in_nover,
    out_ver=out_ver,
)

# Print the shapes of xtrain and ytrain
print(f"xtrain shape: {xtrain.shape}")
print(f"ytrain shape: {ytrain.shape}")


# Prepare dataset and dataloaders
data = MyDataset(X=xtrain, Y=ytrain)
split_data = torch.utils.data.random_split(
    data, [0.75, 0.25], generator=torch.Generator().manual_seed(42)
)
train_dataloader = DataLoader(split_data[0], batch_size=128, shuffle=True)
val_dataloader = DataLoader(split_data[1], batch_size=len(split_data[1]), shuffle=True)

# Model training parameters
learning_rate = 1e-5
epochs = 100
hidden_layers = 8
hidden_size = 500

model = FullyConnected(ilev, in_ver, in_nover, out_ver, hidden_layers, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
early_stopper = early_stopping(patience=5, min_delta=0)

# Train the model with early stopping
train_losses, val_losses = train_with_early_stopping(
    train_dataloader,
    val_dataloader,
    model,
    optimizer,
    criterion,
    early_stopper,
    epochs=epochs,
)
print(f"Train Loss: {train_losses}")
print(f"Valid Loss: {val_losses}")

# Save the trained model
torch.save(model.state_dict(), trained_model_path)

# Load the trained model for prediction
model.load_state_dict(torch.load(trained_model_path))
model.eval()
print()

# Prepare input data for prediction
# For prediction, we need new input data. Here, we use different files for simplicity.
test_data = load_variables(directory_path, features, 4, 5)
normalized_test_data = normalize_data(test_data, mean_dict, std_dict)
x_test, y_test = data_loader(
    features,
    normalized_test_data,
    ilev=ilev,
    in_ver=in_ver,
    in_nover=in_nover,
    out_ver=out_ver,
)

# Convert test data to tensors
x_test_tensor = torch.tensor(x_test, dtype=torch.float64).T

# Make predictions
with torch.no_grad():
    predictions = model(x_test_tensor).numpy()

# Print predictions
print("Predictions Shape:\n", predictions.shape)
# print("Predictions:\n", predictions)

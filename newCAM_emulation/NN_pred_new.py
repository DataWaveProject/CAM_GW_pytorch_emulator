"""Prediction module for the neural network."""

import matplotlib.pyplot as plt
import Model
import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as nnF
import torchvision
from loaddata import data_loader, newnorm
# from savedata import save_netcdf_file
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

"""
Determine if any GPUs are available
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


"""
Initialize Hyperparameters
"""
ilev = 93
ilev_94 = 94

dim_NN = 8 * ilev + 4
dim_NNout = 2 * ilev

batch_size = 8
learning_rate = 1e-4
num_epochs = 1



"""
Initialize the network and the Adam optimizer
"""
GWnet = Model.FullyConnected()

optimizer = torch.optim.Adam(GWnet.parameters(), lr=learning_rate)

model_path = "/glade/derecho/scratch/yqsun/archive/meanstd/saved_convNN_SCM.pt"

data_vars = []

# GWnet.load_state_dict(torch.load("./conv_torch.pth"))
GWnet = torch.jit.load(model_path)
GWnet.eval()

# Load the .npz file
npz_file = np.load('training_data.npz')

# List all variables stored in the .npz file
print("Variables in the .npz file:")
print(npz_file.files)

# Access each variable
U = npz_file['U']
V = npz_file['V']
T = npz_file['T']
Z3 = npz_file['Z3']
RHOI = npz_file['RHOI']
DSE = npz_file['DSE']
NMBV = npz_file['NMBV']
NETDT = npz_file['NETDT']
PS = npz_file['PS']
lat = npz_file['lat']
lon = npz_file['lon']
UTGWSPEC = npz_file['UTGWSPEC']
VTGWSPEC = npz_file['VTGWSPEC']

# Print shapes or inspect the loaded variables
print(f"U shape: {U.shape}")
print(f"U shape: {U.size}")
print(f"V shape: {V.shape}")
print(f"T shape: {T.shape}")
print(f"Z3 shape: {Z3.shape}")
print(f"Lat shape: {lat.shape}")
print(f"Lon shape: {lon.shape}")
print(f"PS shape: {PS.shape}")
print(f"RHOI shape: {RHOI.shape}")


time_dim, ilev_dim, column_dim = U.shape
print(f"Data shape (time, levels, columns): {U.shape}")

#Loop over timesteps and columns
for t in range(time_dim):  # Iterate over time
    for col in range(column_dim):  # Iterate over columns (latitude/longitude)
        # print(f"Processing time {t}, column {col}")
        
        # Extract all levels for the current time and column
        U_sc = U[t, :, col]  # Shape: (ilev,)
        V_sc = V[t, :, col]  # Shape: (ilev,)
        T_sc = T[t, :, col]  # Shape: (ilev,)
        Z3_sc = Z3[t, :, col]  # Shape: (ilev,)
        RHOI_sc = RHOI[t, :, col]  # Shape: (ilev_94,) for RHOI
        DSE_sc = DSE[t, :, col]  # Shape: (ilev,)
        NMBV_sc = NMBV[t, :, col]  # Shape: (ilev,)
        NETDT_sc = NETDT[t, :, col]  # Shape: (ilev,)
        PS_sc = PS[t, col]  # Shape: (1,) for pressure surface

        lat_sc = lat[col]  # Latitude for the column
        lon_sc = lon[col]  # Longitude for the column
        UTGWSPEC_sc = UTGWSPEC[t, :, col]  # Shape: (ilev,)
        VTGWSPEC_sc = VTGWSPEC[t, :, col]  # Shape: (ilev,)

        ##Call the data_loader for the current slices
        x_test, y_test = data_loader(U_sc, V_sc, T_sc, DSE_sc, NMBV_sc, NETDT_sc, Z3_sc, RHOI_sc, PS_sc, lat_sc, lon_sc, UTGWSPEC_sc, VTGWSPEC_sc)
        # print(f'xtest: {x_test.shape}')
        # print(f'ytest: {y_test.shape}')
        
        data = Model.myDataset(X=x_test, Y=y_test)
        # test_loader = DataLoader(data, batch_size=len(data), shuffle=False)
        # print(test_loader)
        
        
        
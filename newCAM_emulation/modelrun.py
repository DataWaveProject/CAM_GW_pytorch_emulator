import torch
from torch import nn
import numpy as np

import netCDF4 as nc

# Path to the .nc file
file_path = 'Demodata/Convection/newCAM_demo_sub_5.nc'


# Open the netCDF file
with nc.Dataset(file_path, "r") as f:
    # List all variables in the file along with their shapes
    print("Variables in the file:")
    for var_name, var in f.variables.items():
        print(f"Variable '{var_name}' shape: {var.shape}")



# # import numpy as np

# def get_variable_names(npz_file_path):
#     # Load the .npz file
#     npz_data = np.load(npz_file_path)

#     # Get the list of variable names
#     variable_names = list(npz_data.keys())

#     return variable_names

# # # Example usage:
# # npz_file_path = 'Demodata/Convection/std_demo_sub.npz' # Replace 'your_file_path.npz' with the path to your .npz file
# # variable_names = get_variable_names(npz_file_path)
# # print("Variable names:", variable_names)

"""Implementing data loader for training neural network."""

import os
import re

import netCDF4 as nc
import numpy as np
import torch

# ruff: noqa: PLR0913
# ruff: noqa: PLR2004

def load_variables(directory_path, variable_names, startfile, endfile):
    """
    Load specified variables from NetCDF files in the given directory.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing NetCDF files.
    variable_names : list of str
        List of variable names to load.
    startfile : int
        Starting file number.
    endfile : int
        Ending file number.

    Returns
    -------
    dict
        Dictionary containing loaded variables data.
    """
    variable_mapping = {"NM": "NMBV"}
    variable_data = {}
    pattern = re.compile(r"^newCAM_demo_sub_\d{startfile,endfile}$")

    for file_name in os.listdir(directory_path):
        if file_name.startswith("newCAM_demo_sub_"):
            file_path = os.path.join(directory_path, file_name)
            with nc.Dataset(file_path) as dataset:
                for var_name in variable_names:
                    mapped_name = variable_mapping.get(var_name, var_name)
                    if mapped_name in dataset.variables:
                        var_data = dataset[mapped_name][:]
                        variable_data[var_name] = var_data

    return variable_data


def load_mean_std(file_path_mean, file_path_std, variable_names):
    """
    Load mean and standard deviation values for specified variables from files.

    Parameters
    ----------
    file_path_mean : str
        Path to the file containing mean values.
    file_path_std : str
        Path to the file containing standard deviation values.
    variable_names : list of str
        List of variable names.

    Returns
    -------
    tuple of dict
        Dictionaries containing mean and standard deviation values.
    """
    mean_data = np.load(file_path_mean)
    std_data = np.load(file_path_std)
    mean_dict = {var_name: mean_data[var_name] for var_name in variable_names}
    std_dict = {var_name: std_data[var_name] for var_name in variable_names}
    return mean_dict, std_dict


def normalize_data(variable_data, mean_values, std_values):
    """
    Normalize the data using mean and standard deviation values.

    Parameters
    ----------
    variable_data : dict
        Dictionary containing the variable data.
    mean_values : dict
        Dictionary containing mean values.
    std_values : dict
        Dictionary containing standard deviation values.

    Returns
    -------
    dict
        Dictionary containing normalized data.
    """
    normalized_data = {}
    for var_name, var_data in variable_data.items():
        if var_name in mean_values and var_name in std_values:
            mean = mean_values[var_name]
            std = std_values[var_name]
            normalized_var_data = (var_data - mean) / std
            normalized_data[var_name] = normalized_var_data
    return normalized_data


def data_loader(variable_names, normalized_data, ilev, in_ver, in_nover, out_ver):
    """
    Prepare the data for training by organizing it into input and output arrays.

    Parameters
    ----------
    variable_names : list of str
        List of variable names.
    normalized_data : dict
        Dictionary containing normalized data.
    ilev : int
        Number of vertical levels.
    in_ver : int
        Number of input variables that vary across vertical levels.
    in_nover : int
        Number of input variables that do not vary across vertical levels.
    out_ver : int
        Number of output variables that vary across vertical levels.


    Returns
    -------
    tuple of np.ndarray
        Input and output arrays for training.
    """
    Ncol = normalized_data[variable_names[1]].shape[2]
    dim_NN = int(in_ver * ilev + in_nover)
    dim_NNout = int(out_ver * ilev)
    x_train = np.zeros([dim_NN, Ncol])
    y_train = np.zeros([dim_NNout, Ncol])
    target_var = ["UTGWSPEC", "VTGWSPEC"]
    y_index = 0
    x_index = 0
    for var_name, var_data in normalized_data.items():
        var_shape = var_data.shape
        if var_name in target_var:
            y_train[y_index * ilev : (y_index + 1) * ilev, :] = var_data
            y_index += 1
        elif len(var_shape) == 2:
            x_train[x_index, :] = var_data
        elif len(var_shape) == 3:
            new_ilev = var_shape[1]
            x_train[x_index : x_index + new_ilev, :] = var_data
        x_index += 1
    return x_train, y_train


class MyDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for loading features and labels.

    Parameters
    ----------
    X : np.ndarray
        Feature data.
    Y : np.ndarray
        Label data.

    Attributes
    ----------
    features : torch.Tensor
        Tensor containing the feature data.
    labels : torch.Tensor
        Tensor containing the label data.
    """

    def __init__(self, X, Y):
        self.features = torch.tensor(X, dtype=torch.float64)
        self.labels = torch.tensor(Y, dtype=torch.float64)

    def __len__(self):
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.features.T)

    def __getitem__(self, idx):
        """
        Return a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple of torch.Tensor
            Feature and label tensors for the sample.
        """
        feature = self.features[:, idx]
        label = self.labels[:, idx]
        return feature, label

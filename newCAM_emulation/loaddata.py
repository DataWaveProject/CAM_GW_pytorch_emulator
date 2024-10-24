"""Implementing data loader for training neural network."""

import numpy as np

# ruff: noqa: PLR0913
# ruff: noqa: PLR2004

ilev = 93
dim_NN = int(8 * ilev + 4)
dim_NNout = int(2 * ilev)


def newnorm(var, varm, varstd):
    """Normalize the input variable(s) using mean and standard deviation.

    Args:
        var (numpy.ndarray): Input variable(s) to be normalized.
        varm (numpy.ndarray): Mean of the variable(s).
        varstd (numpy.ndarray): Standard deviation of the variable(s).

    Returns
    -------
        numpy.ndarray: Normalized variable(s).
    """
    
    print(f'Var shape - {var.shape}')
    print(f'Varm shape - {varm.shape}')
    print(f'Varstd shape - {varstd.shape}')
    
    var = var[:,0]
    varm = np.array(varm)
    varstd = np.array(varstd)
    
    dim = varm.size
    
    # if dim > 1:
    #     vara = var - varm[:, :]
    #     varstdmax = varstd.copy()
    #     varstdmax[varstd == 0.0] = 1.0
    #     tmp = vara / varstdmax[:, :]
    if dim == 1:
        # var = var[:,0] 
        vara = var - varm[:]
        varstdmax = varstd.copy()
        varstdmax[varstd == 0.0] = 1.0
        tmp = vara / varstdmax[:]
    else:
        tmp = (var - varm) / varstd
    return tmp





def data_loader(U, V, T, DSE, NM, NETDT, Z3, RHOI, PS, lat, lon, UTGWSPEC, VTGWSPEC):
    """
    Load and preprocess input data for neural network training.

    Args:
        U (numpy.ndarray): Zonal wind component.
        V (numpy.ndarray): Meridional wind component.
        T (numpy.ndarray): Temperature.
        DSE (numpy.ndarray): Dry static energy.
        NM (numpy.ndarray): Northward mass flux.
        NETDT (numpy.ndarray): Net downward total radiation flux.
        Z3 (numpy.ndarray): Geopotential height.
        RHOI (numpy.ndarray): Air density.
        PS (numpy.ndarray): Surface pressure.
        lat (numpy.ndarray): Latitude.
        lon (numpy.ndarray): Longitude.
        UTGWSPEC (numpy.ndarray): Target zonal wind spectral component.
        VTGWSPEC (numpy.ndarray): Target meridional wind spectral component.

    Returns
    -------
        tuple: A tuple containing the input data and target data arrays.
    """
    # print(f'U-shape in Dataloader{U.shape}')
    # Ncol = U.shape[1] 
    # Nlon = U.shape[2]
    # Ncol = Nlat*Nlon
    # print (ilev)
    # print(Ncol)
    
    x_train = np.zeros([dim_NN])
    y_train = np.zeros([dim_NNout])
    
    # x_train[0:ilev, :] = U.reshape(ilev, Ncol)
    # x_train[ilev : 2 * ilev, :] = V.reshape(ilev, Ncol)
    # x_train[2 * ilev : 3 * ilev, :] = T.reshape(ilev, Ncol)
    # x_train[3 * ilev : 4 * ilev, :] = DSE.reshape(ilev, Ncol)
    # x_train[4 * ilev : 5 * ilev, :] = NM.reshape(ilev, Ncol)
    # x_train[5 * ilev : 6 * ilev, :] = NETDT.reshape(ilev, Ncol)
    # x_train[6 * ilev : 7 * ilev, :] = Z3.reshape(ilev, Ncol)
    # x_train[7 * ilev : 8 * ilev + 1, :] = RHOI.reshape(ilev + 1, Ncol)
    # x_train[8 * ilev + 1 : 8 * ilev + 2, :] = PS.reshape(1, Ncol)
    # x_train[8 * ilev + 2 : 8 * ilev + 3, :] = lat.reshape(1, Ncol)
    # x_train[8 * ilev + 3 : ilev * ilev + 4, :] = lon.reshape(1, Ncol)

    # y_train[0:ilev, :] = UTGWSPEC.reshape(ilev, Ncol)
    # y_train[ilev : 2 * ilev, :] = VTGWSPEC.reshape(ilev, Ncol)
    
    # Populate the input data (x_train)
    x_train[0:ilev] = U  # U: shape (ilev,)
    x_train[ilev:2 * ilev] = V
    x_train[2 * ilev:3 * ilev] = T
    x_train[3 * ilev:4 * ilev] = DSE
    x_train[4 * ilev:5 * ilev] = NM
    x_train[5 * ilev:6 * ilev] = NETDT
    x_train[6 * ilev:7 * ilev] = Z3
    x_train[7 * ilev:8 * ilev] = RHOI[:ilev]  # First 93 levels of RHOI

    # Handle the extra 94th level of RHOI
    x_train[8 * ilev] = RHOI[ilev]  # The 94th level of RHOI (extra level)

    # Add scalar values for PS, latitude, and longitude
    x_train[8 * ilev + 1] = PS  # Surface pressure (scalar)
    x_train[8 * ilev + 2] = lat  # Latitude (scalar)
    x_train[8 * ilev + 3] = lon  # Longitude (scalar)

    # Populate the target data (y_train)
    y_train[0:ilev] = UTGWSPEC
    y_train[ilev:2 * ilev] = VTGWSPEC


    return x_train, y_train

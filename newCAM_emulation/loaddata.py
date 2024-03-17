"""Implementing data loader for training neural network."""

import numpy as np

ilev = 93
dim_NN =int(8*ilev+4)
dim_NNout =int(2*ilev)

def newnorm(var, varm, varstd):
  """Normalizes the input variable(s) using mean and standard deviation.

  Args:
      var (numpy.ndarray): Input variable(s) to be normalized.
      varm (numpy.ndarray): Mean of the variable(s).
      varstd (numpy.ndarray): Standard deviation of the variable(s).

  Returns
  -------
      numpy.ndarray: Normalized variable(s).
  """
  dim=varm.size
  if dim > 1 :
    vara = var - varm[:, :]
    varstdmax = varstd
    varstdmax[varstd==0.0] = 1.0
    tmp = vara / varstdmax[:, :]
  else:
    tmp = ( var - varm ) / varstd
  return tmp


def data_loader (U,V,T, DSE, NM, NETDT, Z3, RHOI, PS, lat, lon, UTGWSPEC, VTGWSPEC):
  """
  Loads and preprocesses input data for neural network training.

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
  Ncol = U.shape[1]
  #Nlon = U.shape[2]
  #Ncol = Nlat*Nlon

  x_train = np.zeros([dim_NN,Ncol])
  y_train = np.zeros([dim_NNout,Ncol])


  x_train [0:ilev, : ] = U.reshape(ilev, Ncol)
  x_train [ilev:2*ilev, :] = V.reshape(ilev, Ncol)
  x_train [2*ilev:3*ilev,:] = T.reshape(ilev, Ncol)
  x_train [3*ilev:4*ilev, :] = DSE.reshape(ilev, Ncol)
  x_train [4*ilev:5*ilev, :] = NM.reshape(ilev, Ncol)
  x_train [5*ilev:6*ilev, :] = NETDT.reshape(ilev, Ncol)
  x_train [6*ilev:7*ilev, :] = Z3.reshape(ilev, Ncol)
  x_train [7*ilev:8*ilev+1, :] = RHOI.reshape(ilev+1, Ncol)
  x_train [8*ilev+1:8*ilev+2, :] = PS.reshape(1, Ncol)
  x_train [8*ilev+2:8*ilev+3, :] = lat.reshape(1, Ncol)
  x_train [8*ilev+3:ilev*ilev+4, :] = lon.reshape(1, Ncol)

  y_train [0:ilev, :] = UTGWSPEC.reshape(ilev, Ncol)
  y_train [ilev:2*ilev, :] = VTGWSPEC.reshape(ilev, Ncol)

  return x_train,y_train


"""Read the data and the corresponding mean and std deviation"""
"""Iterating through the data files"""
s_list = list(range(1, 6))

for iter in s_list:
    filename = "Demodata/Convection/newCAM_demo_sub_" + str(iter).zfill(1) + ".nc"  # data file
    print('working on: ', filename)
    fm = np.load('Demodata/mean_demo_sub.npz')  # mean file
    fs = np.load('Demodata/std_demo_sub.npz')   # std deviation file


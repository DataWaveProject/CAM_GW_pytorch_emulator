# this code only works if the mean and std are designed to be SCM.

import numpy as np
import netCDF4 as nc

#F = nc.Dataset('CAM_output.nc')
F = nc.Dataset('please enter the CAM output file here')
U = np.asarray(F['U'])
V = np.asarray(F['V'])
T = np.asarray(F['T'])
Z3 = np.asarray(F['Z3'])
RHOI = np.asarray(F['RHOI'])
NETDT = np.asarray(F['NETDT'])
DSE = np.asarray(F['DSE'])
NMBV = np.asarray(F['NMBV'])
lat = np.asarray(F['lat'])
lon = np.asarray(F['lon'])
PS = np.asarray(F['PS'])
###
UTGWSPEC=np.asarray(F['BUTGWSPEC'])
VTGWSPEC=np.asarray(F['BVTGWSPEC'])
print(np.shape(U))


## read in mean and std value for each variable
Fm = nc.Dataset('please enter location of the mean file')
Fs = nc.Dataset('please enter location of the std nc file')
#Fm = nc.Dataset('mid-top-CAM_mean_SCM.nc')
#Fs = nc.Dataset('mid-top-CAM_std_SCM.nc')
Um = np.asarray(Fm['U'])
Vm = np.asarray(Fm['V'])
Tm = np.asarray(Fm['T'])
Z3m = np.asarray(Fm['Z3'])
RHOIm = np.asarray(Fm['RHOI'])
NETDTm = np.asarray(Fm['NETDT'])
DSEm = np.asarray(Fm['DSE'])
NMBVm = np.asarray(Fm['NMBV'])
latm = np.asarray(Fm['lat'])
lonm = np.asarray(Fm['lon'])
PSm = np.asarray(Fm['PS'])
###
UTGWSPECm=np.asarray(Fm['UTGWSPEC'])
VTGWSPECm=np.asarray(Fm['VTGWSPEC'])

Us = np.asarray(Fs['U'])
Vs = np.asarray(Fs['V'])
Ts = np.asarray(Fs['T'])
Z3s = np.asarray(Fs['Z3'])
RHOIs = np.asarray(Fs['RHOI'])
NETDTs = np.asarray(Fs['NETDT'])
DSEs = np.asarray(Fs['DSE'])
NMBVs = np.asarray(Fs['NMBV'])
lats = np.asarray(Fs['lat'])
lons = np.asarray(Fs['lon'])
PSs = np.asarray(Fs['PS'])
###
UTGWSPECs=np.asarray(Fs['UTGWSPEC'])
VTGWSPECs=np.asarray(Fs['VTGWSPEC'])

ncol = U.shape[2]
print(ncol) # check number of column here, it shall be 48600


Um3d = Um.reshape(1, len(Um), 1)
Vm3d = Vm.reshape(1, len(Um), 1)
Tm3d = Tm.reshape(1, len(Um), 1)
Z3m3d = Z3m.reshape(1, len(Um), 1)
RHOIm3d = RHOIm.reshape(1, len(Um)+1, 1)
NETDTm3d = NETDTm.reshape(1, len(Um), 1)
DSEm3d = DSEm.reshape(1, len(Um), 1)
NMBVm3d = NMBVm.reshape(1, len(Um), 1)

Us3d = Us.reshape(1, len(Um), 1)
Vs3d = Vs.reshape(1, len(Um), 1)
Ts3d = Ts.reshape(1, len(Um), 1)
Z3s3d = Z3s.reshape(1, len(Um), 1)
RHOIs3d = RHOIs.reshape(1, len(Um)+1, 1)
NETDTs3d = NETDTs.reshape(1, len(Um), 1)
DSEs3d = DSEs.reshape(1, len(Um), 1)
NMBVs3d = NMBVs.reshape(1, len(Um), 1)

UTGWSPECm3d = UTGWSPECm.reshape(1, len(Um), 1)
VTGWSPECm3d = VTGWSPECm.reshape(1, len(Um), 1)
UTGWSPECs3d = UTGWSPECs.reshape(1, len(Um), 1)
VTGWSPECs3d = VTGWSPECs.reshape(1, len(Um), 1)

# mean and std of  PS, lat, lon would be 1 number, no need to do reshaping.
lat = (lat - latm) / lats
lon = (lon - lonm) / lons
PS = (PS - PSm) / PSs
U = (U - Um3d) / Us3d
V = (V - Vm3d) / Vs3d
T = (T - Tm3d) / Ts3d
Z3 = (Z3 - Z3m3d) / Z3s3d
DSE = (DSE - DSEm3d) / DSEs3d
NMBV = (NMBV - NMBVm3d) / NMBVs3d
RHOI = (RHOI - RHOIm3d) / RHOIs3d
NETDT = (NETDT - NETDTm3d) / NETDTs3d

UTGWSPEC = (UTGWSPEC - UTGWSPECm3d) / UTGWSPECs3d
VTGWSPEC = (VTGWSPEC - VTGWSPECm3d) / VTGWSPECs3d


### save all the normalized value
np.savez('./training_data.npz', U=U, V=V, T=T, Z3=Z3, RHOI=RHOI, DSE=DSE, NMBV=NMBV, NETDT=NETDT, PS=PS, lat=lat, lon=lon, UTGWSPEC=UTGWSPEC, VTGWSPEC=VTGWSPEC)

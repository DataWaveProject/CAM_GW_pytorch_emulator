# newCAM-Emulation
This repo is to build a coupling tool between NN and Fortran.

The example we show here is to couple a NN emultor of the gravity wave drag (GWD, both zonal and meridional ) scheme back to the mid-top-CAM model.
* GWD Emulator: PyTorch
* [mid-top-CAM](https://github.com/ESCOMP/CAM): Fortran code


## mid-top-CAM model ##
More introduction on the NCAR CAM model can be found in the official GitHub page of CAM model.
>https://github.com/ESCOMP/CAM


## GWD emulator 

### training emulator
This emulator is trained offline with data generated from the physics scheme in mid-top-CAM model

More detail on the emulator can be found here in [this paper : arxiv link for now](https://arxiv.org/abs/2311.17078)

### Sample training data from mid-top-CAM.
We provide some demodata here for the training. The original training data is 3D global output from the mid-top CAM model, on the original model grid, with a total size of 2Tb. The demo data here is to illustrate the CAM output only, due to storage limit of Github. NN trained on this Demodata will not work.

```
Variable name in the Demo file:
The same variables are used by the original physics scheme in the CAM model.

U : zonal wind
V : meridional wind 
T : temperature
Z3: geopotential height (above sea level)
DSE: dry static energy
NETDT: net heating rate
NMBV: Brunt Vaisala frequency
RHOI: density at interfaces
PS: surface pressure
lat: latitude
lon: longitude

--
GWD:
UTGWSPEC: (zonal drag)
VTGWSPEC: (meridional drag)


In CAM, different names are used for GWD to distinguish them based on their sources:

  * For convective gravity waves: BUTGWSPEC, BVTGWSPEC
  * For frontal gravity waves: UTGWSPEC, VTGWSPEC
  * For orographic gravity waves: UTGWORO, VTGWORO

```

### NN architechure of the emulator
we are using a basic fully connected neural network for now. CNN and FNO will be added in the future.
<img width="651" alt="Screenshot 2024-02-26 at 7 28 10 AM" src="https://github.com/DataWaveProject/newCAM_emulation/assets/85260799/01c42044-6e1c-4bf2-8b56-1ed999933d15">

# Installing

Clone this repo and enter it.\
Then run:
```
pip install .
```
to install the neccessary dependencies.\
It is recommended this is done from inside a virtual environment.

# data loader
load 3D CAM data and reshaping them to the NN input.

# Using a FNN to train and predict the GWD
train.py train the files and generate the weights for NN.

NN-pred.py load the weights and do prediction.

# Coupling ? future work
replace original GWD scheme in WACCM with this emulator.

a. the emulator can be trained offline

b. training the emulator online



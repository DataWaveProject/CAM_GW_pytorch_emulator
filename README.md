# Overview
The repository contains the code for a machine learning model that emulates the climatic process of gravity wave drag (GWD, both zonal and meridional). The model is a part of parameterization scheme where smaller and highly dynamical climatic processes are emulated using neural networks. 

Gravity waves, also called buyoncy waves are formed due to displacement of air in the atmosphere instigated by differnt physical mechanisms, such as moist convection, orographic lifting, shear unstability etc. These waves can propagate both vertically and horizontally through the lift and drag mechanism respectively. This ML model focuses on the drag component of gravity waves.

The long-term goal of the model is to be coupled with a larger fortran-based numerical weather prediction model called the Mid-top CAM Model (Community Atmospheric Model).  
https://www.cesm.ucar.edu/models/cam.


# Model Architecture
The machine leaning model is a Feed Forward Neural Network (FFNN) with 10 hidden layers and 500 neurons in 
each layer. The activation used at each layer is a Sigmoid Linear Unit (SiLU) activation function.

## Dataset
The dataset is available in the `Demodata`

Input variables: pressure levels, latitude, longitude

Output variables: zonal drag force, meridional drag force




<!-- # newCAM-Emulation
This is a DNN written with PyTorch to Emulate the gravity wave drag (GWD, both zonal and meridional ) in the WACCM Simulation.


# DemoData
Sample output data from CAM.
It is 3D global output from the mid-top CAM model, on the original model grid.

However, the demo data here is one very small part of the CAM output due to storage limit of Github. NN trained on this Demodata will not work.

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

b. training the emulator online -->


https://arxiv.org/pdf/2311.17078.pdf
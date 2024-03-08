# newCAM-Emulation

This is a DNN written with PyTorch to Emulate the gravity wave drag (GWD, both zonal and meridional) in the CAM model.
The repository contains the code for a machine learning model that emulates the climatic process of gravity wave drag (GWD, both zonal and meridional).
The model is a part of parameterization scheme where smaller and highly dynamical climatic processes are emulated using neural networks. 

Gravity waves, also called buyoncy waves are formed due to displacement of air in the atmosphere instigated by differnt physical mechanisms, such as moist convection, orographic lifting, shear unstability etc. These waves can propagate both vertically and horizontally through the lift and drag mechanism respectively. This ML model focuses on the drag component of gravity waves.

The long-term goal of the model is to be coupled with a larger fortran-based numerical weather prediction model called the Mid-top CAM Model (Community Atmospheric Model).  
https://www.cesm.ucar.edu/models/cam.

# Installing
1. Change your current working directory to the location where you want to clone the repository
   ```bash
    git clone git@github.com:DataWaveProject/newCAM_emulation.git
    ```
    to clone via ssh, or  
    ```bash
    git clone https://github.com/DataWaveProject/newCAM_emulation.git
    ```
    to clone via https
2. Then run below command to install the neccessary dependencies:
    ```
    pip install .
    ```
It is recommended this is done from inside a virtual environment.


# Model Description

## Architecture
The machine leaning model is a Feed Forward Neural Network (FFNN) with 10 hidden layers and 500 neurons in 
each layer. The activation used at each layer is a Sigmoid Linear Unit (SiLU) activation function.

## Dataset
The dataset available in the `Demodata` is a sample output data from CAM. It is 3D global output from the mid-top CAM model, on the original model grid. The demo data here is one very small part of the CAM output and is only for demo purpose.

- Input variables: pressure levels, latitude, longitude

- Output variables: zonal drag force, meridional drag force

The data has been split in a ratio of 75:25 into training and validation sets. The input variables have been normalised using mean and standard deviation before feeding them to the model for training. Normalisation allows all the inputs to have similar ranges and distribution, hence preventing variables wiht large numerical scale to dominate the predictions.

## Training
The model is trained using the script `train.py` using the demo data. The optimiser used is an `Adam` optimiser with a `learning rate` of 0.001. The data is divided into 128 batches for faster training and effcient memory usage and is run on the model for 100 `epochs`. The training comprises of an `early stopping` mechanism that helps prevent overfitting of the model. The loss in making the predictions is quantified in the form of an `MSE` (mean squared error). The  

## Repository Layout
The `Demodata` folder contains the demo data used to train and test the model

The `newCAM_emulation` folder contains the code that is required to load data, train the model and make predictions which is structured as following:
> `train.py` - train the model

> `NN-pred.py` - predict the GWD using the trained model
    
> `loaddata.py` - load the data and reshape it to the NN input

> `model.py` - define the NN model

## Usage Instructions
To use the repository, following steps are required:
1. For example, to run the `train.py` script to train the model, run the below command: 
    ```bash
    python3 train.py
    ```

### Reference Paper:

**Data Imbalance, Uncertainty Quantification, and Generalization via Transfer Learning in Data-driven Parameterizations: Lessons from the Emulation of Gravity Wave Momentum Transport in WACCM.** 

 *Authors: Y. Qiang Sun and Hamid A. Pahlavan and Ashesh Chattopadhyay and Pedram Hassanzadeh and Sandro W. Lubis and M. Joan Alexander and Edwin Gerber and Aditi Sheshadri and Yifei Guan*
https://arxiv.org/pdf/2311.17078.pdf

### License:
The repository is licensed under MIT License - see the [LICENSE](LICENSE) file for details.
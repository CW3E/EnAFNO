################ LICENSE ######################################
# This software is Copyright © 2024 The Regents of the University of California.
# All Rights Reserved. Permission to copy, modify, and distribute this software and its documentation
# for educational, research and non-profit purposes, without fee, and without a written agreement is
# hereby granted, provided that the above copyright notice, this paragraph and the following three paragraphs
# appear in all copies. Permission to make commercial use of this software may be obtained by contacting:
#
# Office of Innovation and Commercialization 9500 Gilman Drive, Mail Code 0910 University of California La Jolla, CA 92093-0910 innovation@ucsd.edu
# This software program and documentation are copyrighted by The Regents of the University of California. The software program and documentation are
# supplied “as is”, without any accompanying services from The Regents. The Regents does not warrant that the operation of the program will
# be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.
#
# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER
# IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
################################################################
#################### Import libraries ##########################
## Import libraries
import os
import numpy as np
import xarray as xr
import dask as da
import pandas as pd
import torch # https://varhowto.com/install-pytorch-cuda-10-1/  (with pip)
import torch.distributed as dist  # Good resource!!: https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
from torch.nn.parallel import DistributedDataParallel
import torch.cuda.amp as amp
import gc
#################### Define setup parameters ##################
## Set the working directory
workdir='/your_working_directory/'
os.chdir(workdir)
for file in os.listdir(workdir+'/utils/'):
    if file.endswith('.py') and file.strip('._')==file:
        exec(open(workdir+'/utils/'+file).read())
#################### Define setup parameters ##################
nwm='afno-a-1' # there are three branches ("a","b","c") and 1,2,...,30 models per branch
number_of_members=6 # We have 6 bred vectors per NWM
date_ic=np.datetime64('2018-04-04T00:00:00') # Initial condition
lead_time=28 # forecast lead time
afno_params={'patch_size': 8,
             'number_of_afno_blocks': 8,
             'vars': ['ua500','ua850','ua1000',
                      'va500','va850','va1000',
                      'z50','z500','z850','z1000',
                      'ta500','ta850',
                      'hur500','hur850',
                      'uas', 'vas', 'tas', 'sp', 'mslp','tcwv']}
bred_params={'sign': 'both', 'k': 0.15}
# In this notebook we assume that the 90 AFNO models (or a single illustrative model as example) are stored in the folder: workdir+'/models/'
# In this notebook we assume that the era5 initial conditions are stored in the folder: workdir+'/data/era5/'
#    where each file contains the 20 variables for a particular year, e.g., workdir+'/data/era5/2018.nc'
# In this notebook we assume that the bred vectors are stored in the folder: workdir+'/data/bred_vectors/'
#    where each file contains the 20 variables for a particular year, e.g., workdir+'/data/bred_vectors/...'
################################################################################
# Device
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark=True
device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
##########################################
# Neural weather model
model=AFNONet(afno_params).to(device) # code for this function is available at: Pathak, J., Subramanian, S., Harrington, P., Raja, S., Chattopadhyay, A., Mardani, M., ... & Anandkumar, A. (2022). Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators. arXiv preprint arXiv:2202.11214.
model=load_model(model=model, model_path='./models/'+nwm)
model=model.to(device)
##########################################
# Return mean and std for standardization
data_mean, data_std=load_statistics(path_mean='./data/era5/mean.nc', path_std='./data/era5/std.nc')
##########################################
# Initial condition
data_ic=get_date_ic(date_ic=date_ic, path_data='./data/era5/', vars=afno_params['vars'])
data_ic_raw=data_ic
# Scale data
data_ic=scaleGrid(data_ic, data_mean, data_std)
##########################################
# Create directories to store the predictions
path_date_ic='./data/predictions/'+str(date_ic)
if not os.path.exists(path_date_ic):
    os.mkdir(path_date_ic)
path_ensemble=path_date_ic+'/EnAFNO/'
if not os.path.exists(path_ensemble):
    os.mkdir(path_ensemble)
##########################################
# Loop over the number of members (i.e., bred vectors)
for member in range(number_of_members):
    # Recursive N-step forecast
    pred_list=[]
    for lt in range(lead_time):
        ##########################################
        # Perturb with bred vectors
        if lt==0:
            if bred_params['sign']=='both':
                if member >= 3:
                    sign='negative'
                    member_bred=member-3
                else:
                    sign='positive'
                    member_bred=member
            path_to_bred_vector='./data/bred_vectors/'+str(date_ic)+'/'+nwm+'/member_'+str(member_bred)+'.nc' # see
            data_perturbed=bredNoise(data_ic, path_to_bred_vector=path_to_bred_vector, sign=sign)
        else:
            data_perturbed=scaleGrid(grid=pred_i, data_mean, data_std)
        ##########################################
        # Predict with neural network
        pred_i=predictNWM(grid=data_perturbed, vars=afno_params['vars'], device=device, data_mean=data_mean, data_std=data_std)
        pred_list.append(pred_i)
    ##########################################
    # Concatenate prediction
    pred_iter=xr.concat(pred_list, dim='time')
    ##########################################
    # Fill out metadata and add initial condition
    pred=xr.concat([data_ic_raw, pred_iter], dim='time')
    pred=pred.assign_coords({'member': member})
    ##########################################
    # Save prediction
    pred.to_netcdf(path_ensemble+'member_'+str(member)+'.nc')

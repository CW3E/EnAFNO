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
## Import libraries
import os
import cdsapi
import numpy as np
import xarray as xr

## Set the working directory
workdir='/your_working_directory/'
os.chdir(workdir)

## Define parameters
vars=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure',
        'mean_sea_level_pressure', 'sea_surface_temperature', 'total_column_water_vapour']
years =[str(year) for year in np.arange(1979, 2018+1)]

## Define the dictionary
dict_variab={'sea_surface_temperature': 'sst',
              '10m_u_component_of_wind': 'uas',
              '10m_v_component_of_wind': 'vas',
              'total_precipitation': 'precip',
              '2m_temperature': 'tas',
              'surface_pressure': 'sp',
              'mean_sea_level_pressure': 'mslp',
              'total_column_water_vapour': 'tcwv'}


c=cdsapi.Client()
for var in vars:
    for year in years:
        path='./data/era5/'+dict_variab[var]+'/'
        if  not os.path.exists(path):
            print('Creating directory: '+path)
            os.mkdir(path)
        if not os.path.exists(path+year+'.nc'):
            c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': var,
        'year': year,
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '06:00', '12:00', '18:00',
        ],
    },
    path+'aux.nc')
            grid=xr.open_dataset(path+'aux.nc')
            varName=list(grid.keys())[0]
            grid=grid.rename({varName: dict_variab[var]})
            grid.to_netcdf(path+year+'.nc')
            os.remove(path+'aux.nc')

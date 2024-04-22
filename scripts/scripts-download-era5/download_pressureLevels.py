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
years=[str(year) for year in np.arange(1979, 2018+1)]
pressure_levels=['50', '500', '850', '1000']
var='geopotential'

## Define the dictionary
dict_variab={'geopotential':  'z',
              'temperature':  'ta',
              'specific-humidity':  'hus',
              'relative-humidity':  'hur',
              'u_component_of_wind':  'ua',
              'v_component_of_wind':  'va'}

c=cdsapi.Client()
for pressure_level in pressure_levels:
    for year in years:
        path='./data/era5/'+dict_variab[var]+pressure_level+'/'
        if  not os.path.exists(path):
            print('Creating directory: '+path)
            os.mkdir(path)
        if not os.path.exists(path+year+'.nc'):
            c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': var,
        'pressure_level': pressure_level,
        'year': year,
        'month': [
            '01', '02', '03',
            '04', '05', '06',
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
            grid=grid.rename({varName: dict_variab[var]+pressure_level})
            grid.to_netcdf(path+year+'.nc')
            os.remove(path+'aux.nc')

# Towards calibrated ensembles of neural weather model forecasts
This repository contains the material and guidelines to reproduce the results presented in the manuscript entitled **Towards calibrated ensembles of neural weather models**, submitted to *Journal of Advances in Modeling Earth Systems*. We provide scripts that illustrate how to perform inference with bred vectors for a single AFNO model. The repository is structured in two folders:

* scripts --> Main executable scripts.
  * scripts-inference --> Python scripts to perform inference with a neural weather model by perturbing the initial condition with white noise (`inference_G-AFNO.py`) or bred noise (`inference_EnAFNO.py`). These scripts rely on ERA5 data, a trained AFNO model, and pre-computed bred vectors that can be found in Zenodo: . In this folder, we also provide a python script to compute bred vectors for neural weather models (`compute_bred-vectors.py`). 
  * scripts-download-era5 --> Python scripts to download ERA5 from the Copernicus climate data store. Note that you have to register for an account at the European Center for Medium-Range Weather Forecasts (ECMWF) to download the data.
* utils --> This folder contains the auxiliary scripts that are sourced by the main scripts during execution.

`environment.yml` contains the versions of python necessary to run the scripts of the repository. A conda environment with the appropriate versions can be created by typing:
```
mamba env create -n enafno --file environment.yml
```

# Towards calibrated ensembles of neural weather model forecasts
This repository contains the material and guidelines to reproduce the results presented in the manuscript entitled **Towards calibrated ensembles of neural weather models**, submitted to *Journal of Advances in Modeling Earth Systems*. We provide scripts that illustrate how to perform inference with bred vectors for a single AFNO model. Instructions to train AFNO models are available from [Pathak et al., 2022](https://arxiv.org/pdf/2202.11214.pdf). The repository is structured in two folders:

* scripts --> Main executable scripts.
  * scripts-inference --> Python scripts to perform inference with a neural weather model by perturbing the initial condition with white noise (`inference_G-AFNO.py`) or bred noise (`inference_EnAFNO.py`). We also provide a python script to compute bred vectors for neural weather models (`compute_bred-vectors.py`). 
  * scripts-download-era5 --> Python scripts to download ERA5 from the [Copernicus climate data store](https://cds.climate.copernicus.eu/cdsapp#!/dataset/). Note that you have to [register](https://cds.climate.copernicus.eu/user/login?destination=%2Fcdsapp%23!%2Fhome) for an account at the European Center for Medium-Range Weather Forecasts (ECMWF) to download the data.
* utils --> This folder contains the auxiliary scripts that are sourced by the main scripts during execution.
